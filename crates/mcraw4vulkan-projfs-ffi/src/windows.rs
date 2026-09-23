use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::ffi::{OsStr, OsString, c_void};
use std::fs;
use std::hash::{Hash, Hasher};
use std::iter;
use std::mem;
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::time::{SystemTime, UNIX_EPOCH};

use windows_sys::Win32::Foundation::{
    CloseHandle, E_FAIL, E_INVALIDARG, E_OUTOFMEMORY, ERROR_ACCESS_DENIED, ERROR_FILE_NOT_FOUND,
    ERROR_INSUFFICIENT_BUFFER, ERROR_NOT_SUPPORTED, S_OK,
};
use windows_sys::Win32::Storage::ProjectedFileSystem::{
    PRJ_CALLBACK_DATA, PRJ_CALLBACKS, PRJ_CB_DATA_FLAG_ENUM_RESTART_SCAN,
    PRJ_CB_DATA_FLAG_ENUM_RETURN_SINGLE_ENTRY, PRJ_DIR_ENTRY_BUFFER_HANDLE, PRJ_FILE_BASIC_INFO,
    PRJ_NAMESPACE_VIRTUALIZATION_CONTEXT, PRJ_NOTIFICATION, PRJ_NOTIFICATION_PARAMETERS,
    PRJ_PLACEHOLDER_INFO, PRJ_STARTVIRTUALIZING_OPTIONS, PrjAllocateAlignedBuffer,
    PrjDoesNameContainWildCards, PrjFileNameCompare, PrjFileNameMatch, PrjFillDirEntryBuffer,
    PrjFreeAlignedBuffer, PrjMarkDirectoryAsPlaceholder, PrjStartVirtualizing, PrjStopVirtualizing,
    PrjWriteFileData, PrjWritePlaceholderInfo,
};
use windows_sys::Win32::System::Threading::{
    GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows_sys::core::{GUID, HRESULT, PCWSTR};

use crate::{
    DirectoryEntry, FileKind, PlaceholderInfo, ProjectionInstanceId, ProjectionProvider,
    ProviderError, ProviderResult, StartProjectionError,
};

const MAX_CALLBACK_PATH_U16: usize = 32_768;
const MAX_FILE_DATA_CHUNK_SIZE: usize = 1024 * 1024;
const FILE_ATTRIBUTE_READONLY: u32 = 0x0000_0001;
const FILE_ATTRIBUTE_DIRECTORY: u32 = 0x0000_0010;
const FILE_ATTRIBUTE_ARCHIVE: u32 = 0x0000_0020;
const PROCESS_EXIT_CODE_STILL_ACTIVE: u32 = 259;
static NEXT_INSTANCE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

pub struct Projection {
    root: PathBuf,
    instance_id: ProjectionInstanceId,
    context: PRJ_NAMESPACE_VIRTUALIZATION_CONTEXT,
    stopped: bool,
    _callbacks: Box<PRJ_CALLBACKS>,
    _state: Box<ProjectionState>,
}

impl Projection {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn instance_id(&self) -> ProjectionInstanceId {
        self.instance_id
    }

    pub fn stop(&mut self) {
        self.stop_in_place();
    }

    fn stop_in_place(&mut self) {
        if self.stopped {
            return;
        }

        self.stopped = true;
        self._state.begin_stopping();
        let context = self.context;
        self.context = ptr::null_mut();
        if context.is_null() {
            return;
        }

        // SAFETY: context was returned by a successful PrjStartVirtualizing
        // call for this projection and is stopped at most once. The callback
        // table and state remain owned until native callback teardown completes.
        unsafe {
            PrjStopVirtualizing(context);
        }
        debug_assert_eq!(
            self._state.in_flight_callbacks.load(AtomicOrdering::SeqCst),
            0
        );
    }
}

impl Drop for Projection {
    fn drop(&mut self) {
        self.stop_in_place();
    }
}

struct ProjectionState {
    provider: Box<dyn ProjectionProvider>,
    enumerations: Mutex<HashMap<u128, DirectoryEnumeration>>,
    stopping: AtomicBool,
    in_flight_callbacks: AtomicUsize,
    cancelled_commands: AtomicUsize,
    notifications: AtomicUsize,
}

impl ProjectionState {
    fn new(provider: Box<dyn ProjectionProvider>) -> Self {
        Self {
            provider,
            enumerations: Mutex::new(HashMap::new()),
            stopping: AtomicBool::new(false),
            in_flight_callbacks: AtomicUsize::new(0),
            cancelled_commands: AtomicUsize::new(0),
            notifications: AtomicUsize::new(0),
        }
    }

    fn begin_stopping(&self) {
        self.stopping.store(true, AtomicOrdering::SeqCst);
    }

    fn try_enter_callback(&self) -> Result<CallbackGuard<'_>, HRESULT> {
        if self.stopping.load(AtomicOrdering::SeqCst) {
            return Err(E_FAIL);
        }

        self.in_flight_callbacks
            .fetch_add(1, AtomicOrdering::SeqCst);

        if self.stopping.load(AtomicOrdering::SeqCst) {
            self.in_flight_callbacks
                .fetch_sub(1, AtomicOrdering::SeqCst);
            return Err(E_FAIL);
        }

        Ok(CallbackGuard { state: self })
    }
}

struct CallbackGuard<'a> {
    state: &'a ProjectionState,
}

impl Drop for CallbackGuard<'_> {
    fn drop(&mut self) {
        self.state
            .in_flight_callbacks
            .fetch_sub(1, AtomicOrdering::SeqCst);
    }
}

struct DirectoryEnumeration {
    path: PathBuf,
    search_expression: Option<OsString>,
    entries: Vec<DirectoryEntry>,
    next_index: usize,
    loaded: bool,
    completed: bool,
}

impl DirectoryEnumeration {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            search_expression: None,
            entries: Vec::new(),
            next_index: 0,
            loaded: false,
            completed: false,
        }
    }

    fn load_request(
        &mut self,
        path: &Path,
        search_expression: Option<OsString>,
        restart_scan: bool,
    ) -> Option<(PathBuf, Option<OsString>)> {
        if restart_scan {
            self.path = path.to_path_buf();
            self.search_expression = search_expression;
            self.entries.clear();
            self.next_index = 0;
            self.loaded = false;
            self.completed = false;

            return Some((self.path.clone(), self.search_expression.clone()));
        }

        if !self.loaded {
            self.path = path.to_path_buf();
            self.search_expression = search_expression;
            return Some((self.path.clone(), self.search_expression.clone()));
        }

        None
    }

    fn replace_entries(&mut self, entries: Vec<DirectoryEntry>) {
        self.entries = entries;
        self.next_index = 0;
        self.loaded = true;
        self.completed = self.entries.is_empty();
    }
}

pub fn start_projection<P>(
    root: impl AsRef<Path>,
    provider: P,
) -> Result<Projection, StartProjectionError>
where
    P: ProjectionProvider + 'static,
{
    let root = prepare_empty_root(root.as_ref())?;
    let instance_id = generate_projection_instance_id(&root);
    start_prepared_projection(root, provider, instance_id)
}

pub fn start_projection_with_instance_id<P>(
    root: impl AsRef<Path>,
    provider: P,
    instance_id: ProjectionInstanceId,
) -> Result<Projection, StartProjectionError>
where
    P: ProjectionProvider + 'static,
{
    let root = prepare_empty_root(root.as_ref())?;
    start_prepared_projection(root, provider, instance_id)
}

fn start_prepared_projection<P>(
    root: PathBuf,
    provider: P,
    instance_id: ProjectionInstanceId,
) -> Result<Projection, StartProjectionError>
where
    P: ProjectionProvider + 'static,
{
    let root_utf16 = path_to_null_terminated_utf16(&root);
    let callbacks = Box::new(callbacks());
    let state = Box::new(ProjectionState::new(Box::new(provider)));
    let state_context = state.as_ref() as *const ProjectionState as *const c_void;
    let options = PRJ_STARTVIRTUALIZING_OPTIONS {
        Flags: 0,
        PoolThreadCount: 0,
        ConcurrentThreadCount: 0,
        NotificationMappings: ptr::null_mut(),
        NotificationMappingsCount: 0,
    };

    let instance_guid = guid_from_instance_id(instance_id);

    // SAFETY: root_utf16 is a null-terminated absolute path that remains alive
    // for the duration of the call. Target path and version info are optional
    // and intentionally null. instance_guid is unique to this virtualization
    // root so stale placeholders are not confused with a new provider.
    let mark_result = unsafe {
        PrjMarkDirectoryAsPlaceholder(
            root_utf16.as_ptr(),
            ptr::null(),
            ptr::null(),
            &instance_guid,
        )
    };
    check_hresult("PrjMarkDirectoryAsPlaceholder", mark_result)?;

    let mut context: PRJ_NAMESPACE_VIRTUALIZATION_CONTEXT = ptr::null_mut();

    // SAFETY: root_utf16, callbacks, state_context, options, and context out
    // parameters are valid for the duration of the call. The callbacks and
    // state boxes are stored in Projection and remain alive until after
    // PrjStopVirtualizing has been called.
    let start_result = unsafe {
        PrjStartVirtualizing(
            root_utf16.as_ptr(),
            callbacks.as_ref() as *const PRJ_CALLBACKS,
            state_context,
            &options,
            &mut context,
        )
    };
    check_hresult("PrjStartVirtualizing", start_result)?;

    Ok(Projection {
        root,
        instance_id,
        context,
        stopped: false,
        _callbacks: callbacks,
        _state: state,
    })
}

fn generate_projection_instance_id(root: &Path) -> ProjectionInstanceId {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let counter = u128::from(NEXT_INSTANCE_ID_COUNTER.fetch_add(1, AtomicOrdering::SeqCst));
    let process_id = u128::from(std::process::id());
    let mut hasher = DefaultHasher::new();
    root.as_os_str().hash(&mut hasher);
    let root_hash = u128::from(hasher.finish());

    let mut value = nanos ^ (process_id << 96) ^ (counter << 64) ^ root_hash;
    value &= !(0xfu128 << 76);
    value |= 0x4u128 << 76;
    value &= !(0x3u128 << 62);
    value |= 0x2u128 << 62;

    ProjectionInstanceId::from_u128(value)
}

fn guid_from_instance_id(instance_id: ProjectionInstanceId) -> GUID {
    GUID::from_u128(instance_id.as_u128())
}

pub fn process_id_is_live(process_id: u32) -> bool {
    if process_id == 0 {
        return false;
    }

    // SAFETY: OpenProcess is called with query-only access for a numeric PID.
    // The returned handle, if any, is closed before returning.
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, process_id) };
    if handle.is_null() {
        return false;
    }

    let mut exit_code = 0u32;
    // SAFETY: handle is a valid process handle returned by OpenProcess and
    // exit_code points to writable storage for the duration of the call.
    let ok = unsafe { GetExitCodeProcess(handle, &mut exit_code) };
    // SAFETY: handle is valid and is closed exactly once on this path.
    unsafe {
        CloseHandle(handle);
    }

    ok != 0 && exit_code == PROCESS_EXIT_CODE_STILL_ACTIVE
}

fn prepare_empty_root(root: &Path) -> Result<PathBuf, StartProjectionError> {
    if root.exists() {
        let metadata = fs::metadata(root).map_err(|source| StartProjectionError::Io {
            operation: "metadata",
            path: root.to_path_buf(),
            source,
        })?;
        if !metadata.is_dir() {
            return Err(StartProjectionError::InvalidRoot {
                path: root.to_path_buf(),
                message: "root must be a directory".to_string(),
            });
        }
        let mut entries = fs::read_dir(root).map_err(|source| StartProjectionError::Io {
            operation: "read_dir",
            path: root.to_path_buf(),
            source,
        })?;
        if entries
            .next()
            .transpose()
            .map_err(|source| StartProjectionError::Io {
                operation: "read_dir",
                path: root.to_path_buf(),
                source,
            })?
            .is_some()
        {
            return Err(StartProjectionError::InvalidRoot {
                path: root.to_path_buf(),
                message: "root must be empty".to_string(),
            });
        }
    } else {
        fs::create_dir_all(root).map_err(|source| StartProjectionError::Io {
            operation: "create_dir_all",
            path: root.to_path_buf(),
            source,
        })?;
    }

    fs::canonicalize(root).map_err(|source| StartProjectionError::Io {
        operation: "canonicalize",
        path: root.to_path_buf(),
        source,
    })
}

fn path_to_null_terminated_utf16(path: &Path) -> Vec<u16> {
    path.as_os_str()
        .encode_wide()
        .chain(iter::once(0))
        .collect()
}

fn os_string_to_null_terminated_utf16(value: &OsString) -> Vec<u16> {
    os_str_to_null_terminated_utf16(value.as_os_str())
}

fn os_str_to_null_terminated_utf16(value: &OsStr) -> Vec<u16> {
    value.encode_wide().chain(iter::once(0)).collect()
}

fn callbacks() -> PRJ_CALLBACKS {
    PRJ_CALLBACKS {
        StartDirectoryEnumerationCallback: Some(start_directory_enumeration_callback),
        EndDirectoryEnumerationCallback: Some(end_directory_enumeration_callback),
        GetDirectoryEnumerationCallback: Some(get_directory_enumeration_callback),
        GetPlaceholderInfoCallback: Some(get_placeholder_info_callback),
        GetFileDataCallback: Some(get_file_data_callback),
        QueryFileNameCallback: Some(query_file_name_callback),
        NotificationCallback: Some(notification_callback),
        CancelCommandCallback: Some(cancel_command_callback),
    }
}

fn check_hresult(operation: &'static str, hresult: HRESULT) -> Result<(), StartProjectionError> {
    if hresult >= 0 {
        Ok(())
    } else {
        Err(StartProjectionError::Hresult { operation, hresult })
    }
}

fn hresult_from_win32(error: u32) -> HRESULT {
    if error == 0 {
        S_OK
    } else {
        (0x8007_0000u32 | (error & 0x0000_FFFF)) as HRESULT
    }
}

fn catch_callback(callback: impl FnOnce() -> HRESULT) -> HRESULT {
    catch_unwind(AssertUnwindSafe(callback)).unwrap_or(E_FAIL)
}

fn provider_error_to_hresult(error: ProviderError) -> HRESULT {
    match error {
        ProviderError::NotFound => hresult_from_win32(ERROR_FILE_NOT_FOUND),
        ProviderError::NotAFile => hresult_from_win32(ERROR_ACCESS_DENIED),
        ProviderError::Unsupported => hresult_from_win32(ERROR_NOT_SUPPORTED),
        ProviderError::InvalidPath(_) => E_INVALIDARG,
        ProviderError::Internal(_) => E_FAIL,
    }
}

struct AlignedBuffer {
    buffer: *mut c_void,
}

impl AlignedBuffer {
    fn allocate(
        namespace_context: PRJ_NAMESPACE_VIRTUALIZATION_CONTEXT,
        size: usize,
    ) -> Result<Self, HRESULT> {
        if namespace_context.is_null() || size == 0 {
            return Err(E_INVALIDARG);
        }

        // SAFETY: namespace_context comes from the active ProjFS callback and
        // size is the nonzero number of bytes that will be passed to
        // PrjWriteFileData for this same callback.
        let buffer = unsafe { PrjAllocateAlignedBuffer(namespace_context, size) };
        if buffer.is_null() {
            return Err(E_OUTOFMEMORY);
        }

        Ok(Self { buffer })
    }

    fn copy_from_slice(&self, bytes: &[u8]) {
        // SAFETY: self.buffer was allocated with at least bytes.len() bytes for
        // this write, and the source slice is valid and non-overlapping.
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), self.buffer.cast::<u8>(), bytes.len());
        }
    }

    fn as_ptr(&self) -> *const c_void {
        self.buffer.cast_const()
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: buffer was returned by PrjAllocateAlignedBuffer and is freed
        // exactly once by this RAII guard.
        unsafe {
            PrjFreeAlignedBuffer(self.buffer.cast_const());
        }
    }
}

fn with_projection_state(
    callback_data: *const PRJ_CALLBACK_DATA,
    callback: impl FnOnce(&PRJ_CALLBACK_DATA, &ProjectionState) -> HRESULT,
) -> HRESULT {
    catch_callback(|| {
        if callback_data.is_null() {
            return E_INVALIDARG;
        }

        // SAFETY: callback_data is checked for null above and is valid for the
        // duration of this callback according to the ProjFS callback contract.
        let callback_data = unsafe { &*callback_data };
        let state = callback_data.InstanceContext as *const ProjectionState;
        if state.is_null() {
            return E_INVALIDARG;
        }

        // SAFETY: state is the Box<ProjectionState> pointer passed to
        // PrjStartVirtualizing and Projection keeps it alive until stop
        // returns and all callbacks have completed.
        let state = unsafe { &*state };
        let _callback_guard = match state.try_enter_callback() {
            Ok(guard) => guard,
            Err(hresult) => return hresult,
        };

        callback(callback_data, state)
    })
}

fn guid_key_from_ptr(value: *const GUID) -> Result<u128, HRESULT> {
    if value.is_null() {
        return Err(E_INVALIDARG);
    }

    // SAFETY: value is checked for null above and ProjFS supplies a valid
    // enumeration id pointer for the duration of the callback.
    let guid = unsafe { &*value };

    Ok(u128::from(guid.data1) << 96
        | u128::from(guid.data2) << 80
        | u128::from(guid.data3) << 64
        | u128::from(u64::from_be_bytes(guid.data4)))
}

fn path_from_pcwstr(value: PCWSTR) -> Result<PathBuf, HRESULT> {
    Ok(os_string_from_pcwstr(value)?
        .map(PathBuf::from)
        .unwrap_or_default())
}

fn os_string_from_pcwstr(value: PCWSTR) -> Result<Option<OsString>, HRESULT> {
    if value.is_null() {
        return Ok(None);
    }

    let mut len = 0usize;
    while len < MAX_CALLBACK_PATH_U16 {
        // SAFETY: value is not null and ProjFS provides a null-terminated
        // callback string or search expression. The loop bounds the scan to
        // avoid unbounded reads if an invalid provider string is ever observed.
        let code_unit = unsafe { *value.add(len) };
        if code_unit == 0 {
            // SAFETY: the slice is within the validated prefix before the null
            // terminator read above and remains valid for this callback.
            let slice = unsafe { std::slice::from_raw_parts(value, len) };
            return Ok(Some(OsString::from_wide(slice)));
        }
        len += 1;
    }

    Err(E_INVALIDARG)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SearchExpression {
    All,
    Exact(OsString),
    Wildcard(OsString),
}

impl SearchExpression {
    fn from_raw(search_expression: Option<OsString>) -> Self {
        let Some(pattern) = search_expression else {
            return Self::All;
        };

        if projfs_name_contains_wildcards(pattern.as_os_str()) {
            Self::Wildcard(pattern)
        } else {
            Self::Exact(pattern)
        }
    }

    fn matches(&self, name: &OsStr) -> bool {
        match self {
            Self::All => true,
            Self::Exact(pattern) => {
                compare_projfs_os_strings(name, pattern.as_os_str()) == Ordering::Equal
            }
            Self::Wildcard(pattern) => projfs_name_matches(name, pattern.as_os_str()),
        }
    }
}

fn compare_projfs_os_strings(left: &OsStr, right: &OsStr) -> Ordering {
    let left = os_str_to_null_terminated_utf16(left);
    let right = os_str_to_null_terminated_utf16(right);

    // SAFETY: both UTF-16 buffers are null-terminated and alive for this call.
    let result = unsafe { PrjFileNameCompare(left.as_ptr(), right.as_ptr()) };

    result.cmp(&0)
}

fn projfs_name_contains_wildcards(value: &OsStr) -> bool {
    let value = os_str_to_null_terminated_utf16(value);

    // SAFETY: value is a null-terminated UTF-16 buffer alive for this call.
    unsafe { PrjDoesNameContainWildCards(value.as_ptr()) }
}

fn projfs_name_matches(name: &OsStr, pattern: &OsStr) -> bool {
    let name = os_str_to_null_terminated_utf16(name);
    let pattern = os_str_to_null_terminated_utf16(pattern);

    // SAFETY: both UTF-16 buffers are null-terminated and alive for this call.
    unsafe { PrjFileNameMatch(name.as_ptr(), pattern.as_ptr()) }
}

fn load_directory_entries(
    provider: &dyn ProjectionProvider,
    path: &Path,
    search_expression: Option<OsString>,
) -> ProviderResult<Vec<DirectoryEntry>> {
    let mut entries = provider.list_directory(path)?;
    entries.sort_by(|left, right| {
        compare_projfs_os_strings(left.name.as_os_str(), right.name.as_os_str())
    });

    let search_expression = SearchExpression::from_raw(search_expression);
    entries.retain(|entry| search_expression.matches(entry.name.as_os_str()));

    Ok(entries)
}

fn placeholder_to_basic_info(info: PlaceholderInfo) -> PRJ_FILE_BASIC_INFO {
    let is_directory = info.kind == FileKind::Directory;
    let file_size = if is_directory {
        0
    } else {
        info.byte_len.min(i64::MAX as u64) as i64
    };
    let file_attributes = match info.kind {
        FileKind::Directory => FILE_ATTRIBUTE_DIRECTORY,
        FileKind::RegularFile => FILE_ATTRIBUTE_READONLY | FILE_ATTRIBUTE_ARCHIVE,
    };

    PRJ_FILE_BASIC_INFO {
        IsDirectory: is_directory,
        FileSize: file_size,
        CreationTime: info.creation_time,
        LastAccessTime: info.last_access_time,
        LastWriteTime: info.last_write_time,
        ChangeTime: info.change_time,
        FileAttributes: file_attributes,
    }
}

trait DirectoryEntryWriter {
    fn write_directory_entry(&mut self, entry: &DirectoryEntry) -> HRESULT;
}

struct ProjFsDirectoryEntryWriter {
    handle: PRJ_DIR_ENTRY_BUFFER_HANDLE,
}

impl DirectoryEntryWriter for ProjFsDirectoryEntryWriter {
    fn write_directory_entry(&mut self, entry: &DirectoryEntry) -> HRESULT {
        let basic_info = placeholder_to_basic_info(entry.info);
        let name = os_string_to_null_terminated_utf16(&entry.name);

        // SAFETY: name is null-terminated and alive for the call, basic_info
        // points to a valid PRJ_FILE_BASIC_INFO, and handle was supplied by
        // ProjFS for this callback.
        unsafe { PrjFillDirEntryBuffer(name.as_ptr(), &basic_info, self.handle) }
    }
}

fn fill_directory_entries(
    enumeration: &mut DirectoryEnumeration,
    writer: &mut impl DirectoryEntryWriter,
    return_single_entry: bool,
) -> HRESULT {
    let insufficient_buffer = hresult_from_win32(ERROR_INSUFFICIENT_BUFFER);
    let mut entries_added = 0usize;

    while let Some(entry) = enumeration.entries.get(enumeration.next_index) {
        if entry.name.is_empty() {
            return E_INVALIDARG;
        }

        let result = writer.write_directory_entry(entry);
        if result == insufficient_buffer {
            return if entries_added == 0 {
                insufficient_buffer
            } else {
                S_OK
            };
        }
        if result < 0 {
            return result;
        }

        enumeration.next_index += 1;
        entries_added += 1;

        if return_single_entry {
            if enumeration.next_index >= enumeration.entries.len() {
                enumeration.completed = true;
            }
            return S_OK;
        }
    }

    enumeration.completed = true;
    S_OK
}

// SAFETY: ProjFS supplies callback_data and enumeration_id for this invocation.
// The state guard rejects entry once stopping begins; the GUID and path are
// copied before the callback returns.
unsafe extern "system" fn start_directory_enumeration_callback(
    callback_data: *const PRJ_CALLBACK_DATA,
    enumeration_id: *const GUID,
) -> HRESULT {
    with_projection_state(callback_data, |callback_data, state| {
        let enumeration_key = match guid_key_from_ptr(enumeration_id) {
            Ok(value) => value,
            Err(hresult) => return hresult,
        };
        let path = match path_from_pcwstr(callback_data.FilePathName) {
            Ok(path) => path,
            Err(hresult) => return hresult,
        };
        let mut enumerations = match state.enumerations.lock() {
            Ok(enumerations) => enumerations,
            Err(_) => return E_FAIL,
        };

        enumerations.insert(enumeration_key, DirectoryEnumeration::new(path));
        S_OK
    })
}

// SAFETY: ProjFS supplies callback_data and enumeration_id for this invocation.
// The state guard rejects entry once stopping begins, and no native pointer is
// retained after the GUID key is copied.
unsafe extern "system" fn end_directory_enumeration_callback(
    callback_data: *const PRJ_CALLBACK_DATA,
    enumeration_id: *const GUID,
) -> HRESULT {
    with_projection_state(callback_data, |_callback_data, state| {
        let enumeration_key = match guid_key_from_ptr(enumeration_id) {
            Ok(value) => value,
            Err(hresult) => return hresult,
        };
        let mut enumerations = match state.enumerations.lock() {
            Ok(enumerations) => enumerations,
            Err(_) => return E_FAIL,
        };

        enumerations.remove(&enumeration_key);
        S_OK
    })
}

// SAFETY: ProjFS supplies callback_data, enumeration_id, and a live directory
// buffer for this invocation; search_expression may be null. Strings and the GUID
// are copied, and no native pointer or handle is retained after return.
unsafe extern "system" fn get_directory_enumeration_callback(
    callback_data: *const PRJ_CALLBACK_DATA,
    enumeration_id: *const GUID,
    search_expression: PCWSTR,
    dir_entry_buffer_handle: PRJ_DIR_ENTRY_BUFFER_HANDLE,
) -> HRESULT {
    with_projection_state(callback_data, |callback_data, state| {
        if dir_entry_buffer_handle.is_null() {
            return E_INVALIDARG;
        }

        let enumeration_key = match guid_key_from_ptr(enumeration_id) {
            Ok(value) => value,
            Err(hresult) => return hresult,
        };
        let path = match path_from_pcwstr(callback_data.FilePathName) {
            Ok(path) => path,
            Err(hresult) => return hresult,
        };
        let search_expression = match os_string_from_pcwstr(search_expression) {
            Ok(search_expression) => search_expression,
            Err(hresult) => return hresult,
        };
        let restart_scan = callback_data.Flags & PRJ_CB_DATA_FLAG_ENUM_RESTART_SCAN != 0;
        let return_single_entry =
            callback_data.Flags & PRJ_CB_DATA_FLAG_ENUM_RETURN_SINGLE_ENTRY != 0;

        let load_request = {
            let mut enumerations = match state.enumerations.lock() {
                Ok(enumerations) => enumerations,
                Err(_) => return E_FAIL,
            };
            let enumeration = enumerations
                .entry(enumeration_key)
                .or_insert_with(|| DirectoryEnumeration::new(path.clone()));
            enumeration.load_request(&path, search_expression, restart_scan)
        };

        if let Some((path, search_expression)) = load_request {
            let entries = match load_directory_entries(
                state.provider.as_ref(),
                &path,
                search_expression.clone(),
            ) {
                Ok(entries) => entries,
                Err(error) => return provider_error_to_hresult(error),
            };

            let mut enumerations = match state.enumerations.lock() {
                Ok(enumerations) => enumerations,
                Err(_) => return E_FAIL,
            };
            let Some(enumeration) = enumerations.get_mut(&enumeration_key) else {
                return S_OK;
            };
            enumeration.search_expression = search_expression;
            enumeration.replace_entries(entries);
        }

        let mut enumerations = match state.enumerations.lock() {
            Ok(enumerations) => enumerations,
            Err(_) => return E_FAIL,
        };
        let Some(enumeration) = enumerations.get_mut(&enumeration_key) else {
            return S_OK;
        };
        let mut writer = ProjFsDirectoryEntryWriter {
            handle: dir_entry_buffer_handle,
        };

        fill_directory_entries(enumeration, &mut writer, return_single_entry)
    })
}

// SAFETY: ProjFS supplies callback_data and its path and namespace-context
// fields for this invocation. The state guard rejects entry once stopping begins;
// copied paths and provider references do not escape the callback.
unsafe extern "system" fn get_placeholder_info_callback(
    callback_data: *const PRJ_CALLBACK_DATA,
) -> HRESULT {
    with_projection_state(callback_data, |callback_data, state| {
        let path = match path_from_pcwstr(callback_data.FilePathName) {
            Ok(path) => path,
            Err(hresult) => return hresult,
        };
        let info = match state.provider.placeholder_info(&path) {
            Ok(info) => info,
            Err(error) => return provider_error_to_hresult(error),
        };
        let placeholder_info = PRJ_PLACEHOLDER_INFO {
            FileBasicInfo: placeholder_to_basic_info(info),
            ..PRJ_PLACEHOLDER_INFO::default()
        };
        let path = path_to_null_terminated_utf16(&path);

        // SAFETY: NamespaceVirtualizationContext and FilePathName come from the
        // active callback. path is a null-terminated relative provider path
        // alive for the call, and placeholder_info is a valid initialized value.
        unsafe {
            PrjWritePlaceholderInfo(
                callback_data.NamespaceVirtualizationContext,
                path.as_ptr(),
                &placeholder_info,
                mem::size_of_val(&placeholder_info) as u32,
            )
        }
    })
}

// SAFETY: ProjFS supplies callback_data and its path, namespace-context, and
// data-stream fields for this invocation. Guarded state access and aligned write
// buffers end before the callback returns.
unsafe extern "system" fn get_file_data_callback(
    callback_data: *const PRJ_CALLBACK_DATA,
    byte_offset: u64,
    length: u32,
) -> HRESULT {
    with_projection_state(callback_data, |callback_data, state| {
        let path = match path_from_pcwstr(callback_data.FilePathName) {
            Ok(path) => path,
            Err(hresult) => return hresult,
        };

        write_file_data_from_provider(callback_data, state, &path, byte_offset, length)
    })
}

fn write_file_data_from_provider(
    callback_data: &PRJ_CALLBACK_DATA,
    state: &ProjectionState,
    path: &Path,
    byte_offset: u64,
    length: u32,
) -> HRESULT {
    if length == 0 {
        return S_OK;
    }

    let info = match state.provider.placeholder_info(path) {
        Ok(info) => info,
        Err(error) => return provider_error_to_hresult(error),
    };
    if info.kind != FileKind::RegularFile {
        return provider_error_to_hresult(ProviderError::NotAFile);
    }

    if byte_offset >= info.byte_len {
        return S_OK;
    }

    let mut current_offset = byte_offset;
    let mut remaining = u64::from(length).min(info.byte_len.saturating_sub(byte_offset));

    while remaining > 0 {
        let chunk_len = remaining.min(MAX_FILE_DATA_CHUNK_SIZE as u64) as u32;
        let bytes = match state.provider.read_file(path, current_offset, chunk_len) {
            Ok(bytes) => bytes,
            Err(error) => return provider_error_to_hresult(error),
        };

        if bytes.is_empty() || bytes.len() > chunk_len as usize || bytes.len() as u64 > remaining {
            return E_FAIL;
        }

        let result = write_file_data_chunk(callback_data, current_offset, &bytes);
        if result < 0 {
            return result;
        }

        let bytes_written = bytes.len() as u64;
        current_offset = current_offset.saturating_add(bytes_written);
        remaining = remaining.saturating_sub(bytes_written);

        if bytes_written < u64::from(chunk_len) {
            return S_OK;
        }
    }

    S_OK
}

fn write_file_data_chunk(
    callback_data: &PRJ_CALLBACK_DATA,
    byte_offset: u64,
    bytes: &[u8],
) -> HRESULT {
    if bytes.is_empty() || bytes.len() > u32::MAX as usize {
        return E_INVALIDARG;
    }

    let buffer =
        match AlignedBuffer::allocate(callback_data.NamespaceVirtualizationContext, bytes.len()) {
            Ok(buffer) => buffer,
            Err(hresult) => return hresult,
        };
    buffer.copy_from_slice(bytes);

    // SAFETY: NamespaceVirtualizationContext and DataStreamId come from the
    // active callback. buffer is allocated by ProjFS for this context, contains
    // bytes.len() initialized bytes, and remains alive for the duration of the
    // call. byte_offset and length describe the requested provider range.
    unsafe {
        PrjWriteFileData(
            callback_data.NamespaceVirtualizationContext,
            &callback_data.DataStreamId,
            buffer.as_ptr(),
            byte_offset,
            bytes.len() as u32,
        )
    }
}

// SAFETY: ProjFS supplies callback_data and its path for this invocation. The
// state guard rejects entry once stopping begins, and the copied path retains no
// callback-owned storage.
unsafe extern "system" fn query_file_name_callback(
    callback_data: *const PRJ_CALLBACK_DATA,
) -> HRESULT {
    with_projection_state(callback_data, |callback_data, state| {
        let path = match path_from_pcwstr(callback_data.FilePathName) {
            Ok(path) => path,
            Err(hresult) => return hresult,
        };

        match state.provider.placeholder_info(&path) {
            Ok(_) => S_OK,
            Err(error) => provider_error_to_hresult(error),
        }
    })
}

// SAFETY: ProjFS owns callback_data for this invocation. The remaining native
// pointers are not dereferenced or retained, and state access is teardown-guarded.
unsafe extern "system" fn notification_callback(
    callback_data: *const PRJ_CALLBACK_DATA,
    _is_directory: bool,
    _notification: PRJ_NOTIFICATION,
    _destination_filename: PCWSTR,
    _operation_parameters: *mut PRJ_NOTIFICATION_PARAMETERS,
) -> HRESULT {
    with_projection_state(callback_data, |_callback_data, state| {
        state.notifications.fetch_add(1, AtomicOrdering::SeqCst);
        S_OK
    })
}

// SAFETY: ProjFS owns callback_data for this invocation; guarded state access
// completes synchronously and retains no native pointer.
unsafe extern "system" fn cancel_command_callback(callback_data: *const PRJ_CALLBACK_DATA) {
    let _ = with_projection_state(callback_data, |_callback_data, state| {
        state
            .cancelled_commands
            .fetch_add(1, AtomicOrdering::SeqCst);
        S_OK
    });
}

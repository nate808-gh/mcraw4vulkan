#[cfg(windows)]
fn main() {
    let mut resource = winresource::WindowsResource::new();
    resource.set_icon("../../packaging/windows/mcraw4vulkan.ico");
    resource
        .set("ProductName", "mcraw4vulkan")
        .set("FileDescription", "mcraw4vulkan preflight launcher")
        .set("OriginalFilename", "mcraw4vulkan-preflight-check.exe")
        .set("InternalName", "mcraw4vulkan-preflight-check")
        .set("CompanyName", "nate808-gh")
        .set("LegalCopyright", "Copyright (C) 2026 nate808-gh")
        .set("Comments", "https://github.com/nate808-gh/mcraw4vulkan");
    resource
        .compile()
        .expect("failed to compile Windows resources");
}

#[cfg(not(windows))]
fn main() {}

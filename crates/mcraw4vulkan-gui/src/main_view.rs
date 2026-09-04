use crate::style;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MainLayoutMode {
    Condensed,
    Expanded,
}

impl MainLayoutMode {
    pub fn visible_roles(self) -> &'static [MainLayoutRole] {
        match self {
            MainLayoutMode::Condensed => &CONDENSED_MAIN_LAYOUT_ROLES,
            MainLayoutMode::Expanded => &EXPANDED_MAIN_LAYOUT_ROLES,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MainLayoutRole {
    PrimaryControls,
    MoreOptions,
    DisplayPlayback,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MainLayout {
    pub mode: MainLayoutMode,
    pub primary_controls_width: f32,
    pub more_options_width: f32,
    pub display_width: f32,
    pub outer_margin: f32,
    pub column_gap: f32,
}

impl MainLayout {
    pub fn total_width(self) -> f32 {
        self.outer_margin * 2.0
            + self.primary_controls_width
            + self.more_options_width
            + self.display_width
            + self.column_gap * active_gap_count(self.mode) as f32
    }

    pub fn minimum_required_width(self, minimum_display_width: f32) -> f32 {
        minimum_window_width(self.mode, minimum_display_width)
    }

    pub fn fits_minimum_display_width(self, minimum_display_width: f32) -> bool {
        self.display_width >= non_negative_finite(minimum_display_width)
    }

    pub fn visible_roles(self) -> &'static [MainLayoutRole] {
        self.mode.visible_roles()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaylistEntryVisualState {
    Normal,
    IntendedMounted,
    MountedByGui,
    UnmountedByGui,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaylistEntryTextTone {
    Gray,
    White,
    BrightBlue,
}

pub const BASE_PRIMARY_CONTROLS_WIDTH: f32 = 560.0;
pub const BASE_MORE_OPTIONS_WIDTH: f32 = 360.0;
pub const BUTTON_THREE_CHARACTER_EXTRA_WIDTH: f32 = 56.0;
pub const ORDINARY_BUTTON_HEIGHT: f32 = 34.0;
pub const TRANSPORT_BUTTON_EXTRA_WIDTH: f32 = BUTTON_THREE_CHARACTER_EXTRA_WIDTH;
pub const PRIMARY_CONTROLS_WIDTH: f32 = 728.0;
pub const MORE_OPTIONS_WIDTH: f32 = 472.0;
pub const MAIN_COLUMN_GAP: f32 = 16.0;
pub const MAIN_OUTER_MARGIN: f32 = style::OUTER_MARGIN;
pub const COLUMN_FRAME_INNER_MARGIN: f32 = 18.0;
pub const STANDARD_ITEM_SPACING: f32 = 8.0;
pub const PRIMARY_PLAYLIST_MIN_USEFUL_HEIGHT: f32 = 120.0;
pub const PRIMARY_PLAYLIST_TARGET_FRACTION: f32 = 0.48;
pub const PRIMARY_SECTION_BREAK_HEIGHT: f32 = 25.0;
pub const PRIMARY_SECTION_TITLE_HEIGHT: f32 = 28.0;
pub const PRIMARY_BODY_LINE_HEIGHT: f32 = 28.0;
pub const PRIMARY_STATUS_HEIGHT: f32 = 72.0;
pub const CONTROL_ROW_TOP_PADDING: f32 = STANDARD_ITEM_SPACING;
pub const CONTROL_ROW_BOTTOM_PADDING: f32 = STANDARD_ITEM_SPACING;
pub const FOOTER_ROW_TOP_PADDING: f32 = STANDARD_ITEM_SPACING;
pub const FOOTER_BOTTOM_INSET: f32 = STANDARD_ITEM_SPACING;
pub const PRIMARY_PLAYLIST_HEADER_ACTIONS_HEIGHT: f32 =
    PRIMARY_SECTION_TITLE_HEIGHT + STANDARD_ITEM_SPACING + ORDINARY_BUTTON_HEIGHT;
pub const PRIMARY_PLAYLIST_LIST_TOP_SEPARATOR_HEIGHT: f32 = 25.0;
pub const PRIMARY_PLAYLIST_FIXED_HEIGHT: f32 =
    PRIMARY_PLAYLIST_HEADER_ACTIONS_HEIGHT + PRIMARY_PLAYLIST_LIST_TOP_SEPARATOR_HEIGHT;
pub const PRIMARY_QUICK_PREVIEW_FIXED_HEIGHT: f32 = PRIMARY_SECTION_TITLE_HEIGHT
    + PRIMARY_BODY_LINE_HEIGHT
    + STANDARD_ITEM_SPACING
    + ORDINARY_BUTTON_HEIGHT;
pub const PRIMARY_DNG_FIXED_HEIGHT: f32 = PRIMARY_SECTION_TITLE_HEIGHT
    + PRIMARY_BODY_LINE_HEIGHT * 3.0
    + STANDARD_ITEM_SPACING
    + ORDINARY_BUTTON_HEIGHT
    + 6.0
    + ORDINARY_BUTTON_HEIGHT;
pub const PRIMARY_FOOTER_FIXED_HEIGHT: f32 = ORDINARY_BUTTON_HEIGHT;
pub const SHARED_CONTROL_ROW_GAP: f32 = PRIMARY_SECTION_BREAK_HEIGHT;
pub const MIDDLE_TOP_FIXED_HEIGHT: f32 = PRIMARY_SECTION_TITLE_HEIGHT
    + STANDARD_ITEM_SPACING
    + ORDINARY_BUTTON_HEIGHT
    + 14.0
    + PRIMARY_SECTION_TITLE_HEIGHT
    + PRIMARY_BODY_LINE_HEIGHT
    + STANDARD_ITEM_SPACING
    + ORDINARY_BUTTON_HEIGHT
    + 6.0
    + ORDINARY_BUTTON_HEIGHT;
pub const MIDDLE_QUICK_PREVIEW_FIXED_HEIGHT: f32 = PRIMARY_SECTION_TITLE_HEIGHT
    + STANDARD_ITEM_SPACING
    + ORDINARY_BUTTON_HEIGHT
    + 6.0
    + ORDINARY_BUTTON_HEIGHT;
pub const QUICK_PREVIEW_LEFT_CONTENT_HEIGHT: f32 = PRIMARY_QUICK_PREVIEW_FIXED_HEIGHT;
pub const QUICK_PREVIEW_MIDDLE_CONTENT_HEIGHT: f32 =
    MIDDLE_QUICK_PREVIEW_FIXED_HEIGHT + STANDARD_ITEM_SPACING * 2.0;
pub const QUICK_PREVIEW_CONTENT_HEIGHT: f32 =
    if QUICK_PREVIEW_LEFT_CONTENT_HEIGHT > QUICK_PREVIEW_MIDDLE_CONTENT_HEIGHT {
        QUICK_PREVIEW_LEFT_CONTENT_HEIGHT
    } else {
        QUICK_PREVIEW_MIDDLE_CONTENT_HEIGHT
    };
pub const SHARED_QUICK_PREVIEW_ROW_HEIGHT: f32 =
    CONTROL_ROW_TOP_PADDING + QUICK_PREVIEW_CONTENT_HEIGHT + CONTROL_ROW_BOTTOM_PADDING;
pub const DNG_CONTENT_HEIGHT: f32 = PRIMARY_SECTION_TITLE_HEIGHT
    + PRIMARY_BODY_LINE_HEIGHT * 2.0
    + STANDARD_ITEM_SPACING
    + ORDINARY_BUTTON_HEIGHT
    + STANDARD_ITEM_SPACING * 3.0
    + 6.0
    + ORDINARY_BUTTON_HEIGHT;
pub const SHARED_DNG_ROW_HEIGHT: f32 =
    CONTROL_ROW_TOP_PADDING + DNG_CONTENT_HEIGHT + CONTROL_ROW_BOTTOM_PADDING;
pub const SHARED_FOOTER_ROW_HEIGHT: f32 =
    FOOTER_ROW_TOP_PADDING + ORDINARY_BUTTON_HEIGHT + FOOTER_BOTTOM_INSET;
pub const UNDERSIZED_QUICK_PREVIEW_ROW_HEIGHT: f32 = MIDDLE_QUICK_PREVIEW_FIXED_HEIGHT;
pub const UNDERSIZED_DNG_ROW_HEIGHT: f32 = PRIMARY_SECTION_TITLE_HEIGHT
    + PRIMARY_BODY_LINE_HEIGHT * 2.0
    + STANDARD_ITEM_SPACING
    + ORDINARY_BUTTON_HEIGHT
    + 6.0
    + ORDINARY_BUTTON_HEIGHT;
pub const UNDERSIZED_FOOTER_ROW_HEIGHT: f32 = ORDINARY_BUTTON_HEIGHT;
pub const CONDENSED_MAIN_LAYOUT_ROLES: [MainLayoutRole; 2] = [
    MainLayoutRole::PrimaryControls,
    MainLayoutRole::DisplayPlayback,
];
pub const EXPANDED_MAIN_LAYOUT_ROLES: [MainLayoutRole; 3] = [
    MainLayoutRole::PrimaryControls,
    MainLayoutRole::MoreOptions,
    MainLayoutRole::DisplayPlayback,
];
pub const MAIN_SKELETON_LABELS: &[&str] = &[
    "Primary Controls",
    "No clip playing",
    "Play",
    "Stop",
    "Frame 0",
    "Quick Preview",
    "8 bit medium-quality preview. Toggle F key for full screen preview.",
    "Decoding Settings",
    "Vsync On",
    "Max speed",
    "GPU Decoding",
    "CPU Decoding",
    "Vignette Correction",
    "FPS overlay",
    "Optimized Decode Settings",
    "Compares Default and Offset payload profiles for this system",
    "Run optimizer now",
    "Default Settings",
    "Optimized Settings",
    "DNG",
    "Mount raw video as a full-quality virtual DNG folder",
    "DNG vignette correction avoids the magenta shift of Quick Preview",
    "Playlist",
    "Add files",
    "Remove all",
    "Preview",
    "Remove file",
    "Mount DNG",
    "Mount all DNGs",
    "Unmount DNG",
    "Pipe Example",
    "Unmount all DNGs",
    "No files in playlist.",
];

pub fn target_main_layout(mode: MainLayoutMode, available_width: f32) -> MainLayout {
    let content_width = (non_negative_finite(available_width) - MAIN_OUTER_MARGIN * 2.0).max(0.0);
    let fixed_region_width = fixed_region_width(mode);
    let gap_width = MAIN_COLUMN_GAP * active_gap_count(mode) as f32;
    let display_width = (content_width - fixed_region_width - gap_width).max(0.0);

    MainLayout {
        mode,
        primary_controls_width: PRIMARY_CONTROLS_WIDTH,
        more_options_width: match mode {
            MainLayoutMode::Condensed => 0.0,
            MainLayoutMode::Expanded => MORE_OPTIONS_WIDTH,
        },
        display_width,
        outer_margin: MAIN_OUTER_MARGIN,
        column_gap: MAIN_COLUMN_GAP,
    }
}

pub fn base_primary_three_button_width(item_spacing: f32) -> f32 {
    equal_row_button_width(
        BASE_PRIMARY_CONTROLS_WIDTH - COLUMN_FRAME_INNER_MARGIN * 2.0,
        item_spacing,
        3,
    )
}

pub fn ordinary_primary_three_button_width(item_spacing: f32) -> f32 {
    base_primary_three_button_width(item_spacing) + BUTTON_THREE_CHARACTER_EXTRA_WIDTH
}

pub fn derived_primary_controls_width(item_spacing: f32) -> f32 {
    COLUMN_FRAME_INNER_MARGIN * 2.0
        + ordinary_primary_three_button_width(item_spacing) * 3.0
        + non_negative_finite(item_spacing) * 2.0
}

pub fn base_more_options_two_button_width(item_spacing: f32) -> f32 {
    equal_row_button_width(
        BASE_MORE_OPTIONS_WIDTH - COLUMN_FRAME_INNER_MARGIN * 2.0,
        item_spacing,
        2,
    )
}

pub fn ordinary_more_options_two_button_width(item_spacing: f32) -> f32 {
    base_more_options_two_button_width(item_spacing) + BUTTON_THREE_CHARACTER_EXTRA_WIDTH
}

pub fn derived_more_options_width(item_spacing: f32) -> f32 {
    COLUMN_FRAME_INNER_MARGIN * 2.0
        + ordinary_more_options_two_button_width(item_spacing) * 2.0
        + non_negative_finite(item_spacing)
}

pub fn minimum_window_width(mode: MainLayoutMode, minimum_display_width: f32) -> f32 {
    MAIN_OUTER_MARGIN * 2.0
        + fixed_region_width(mode)
        + MAIN_COLUMN_GAP * active_gap_count(mode) as f32
        + non_negative_finite(minimum_display_width)
}

fn fixed_region_width(mode: MainLayoutMode) -> f32 {
    PRIMARY_CONTROLS_WIDTH
        + match mode {
            MainLayoutMode::Condensed => 0.0,
            MainLayoutMode::Expanded => MORE_OPTIONS_WIDTH,
        }
}

fn active_gap_count(mode: MainLayoutMode) -> usize {
    match mode {
        MainLayoutMode::Condensed => 1,
        MainLayoutMode::Expanded => 2,
    }
}

fn non_negative_finite(value: f32) -> f32 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

pub fn column_content_height(column_box_height: f32) -> f32 {
    (column_box_height - COLUMN_FRAME_INNER_MARGIN * 2.0).max(0.0)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrimaryColumnHeightBudget {
    pub available_height: f32,
    pub fixed_lower_control_height: f32,
    pub playlist_fixed_height: f32,
    pub playlist_list_height: f32,
    pub total_used_height: f32,
    pub outer_scroll_required: bool,
}

impl PrimaryColumnHeightBudget {
    pub fn dng_bottom(&self) -> f32 {
        shared_controls_row_layout(self.available_height).row_bottom(ControlRow::Dng)
    }

    pub fn status_bottom(&self) -> f32 {
        shared_controls_row_layout(self.available_height).row_bottom(ControlRow::Status)
    }

    pub fn footer_bottom(&self) -> f32 {
        shared_controls_row_layout(self.available_height).row_bottom(ControlRow::Footer)
    }
}

pub fn primary_column_height_budget(available_height: f32) -> PrimaryColumnHeightBudget {
    let rows = shared_controls_row_layout(available_height);
    let playlist_fixed_height = PRIMARY_PLAYLIST_FIXED_HEIGHT;

    PrimaryColumnHeightBudget {
        available_height: rows.available_height,
        fixed_lower_control_height: rows.fixed_lower_control_height(),
        playlist_fixed_height,
        playlist_list_height: rows.playlist_list_height,
        total_used_height: rows.total_used_height,
        outer_scroll_required: rows.outer_scroll_required,
    }
}

pub fn primary_fixed_lower_control_height() -> f32 {
    shared_fixed_lower_control_height()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlRow {
    Top,
    QuickPreview,
    Dng,
    Status,
    Footer,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlRowBounds {
    pub top: f32,
    pub height: f32,
}

impl ControlRowBounds {
    pub fn bottom(self) -> f32 {
        self.top + self.height
    }

    pub fn center(self) -> f32 {
        self.top + self.height * 0.5
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SharedControlsRowLayout {
    pub available_height: f32,
    pub top_height: f32,
    pub playlist_list_height: f32,
    pub quick_preview_height: f32,
    pub dng_height: f32,
    pub status_height: f32,
    pub footer_height: f32,
    pub total_used_height: f32,
    pub outer_scroll_required: bool,
}

impl SharedControlsRowLayout {
    pub fn fixed_lower_control_height(self) -> f32 {
        shared_fixed_lower_control_height()
    }

    pub fn row_bounds(self, row: ControlRow) -> ControlRowBounds {
        let top_bottom = self.top_height;
        let quick_top = top_bottom + SHARED_CONTROL_ROW_GAP;
        let quick_bottom = quick_top + self.quick_preview_height;
        let dng_top = quick_bottom + SHARED_CONTROL_ROW_GAP;
        let dng_bottom = dng_top + self.dng_height;
        let status_top = dng_bottom + SHARED_CONTROL_ROW_GAP;
        let status_bottom = status_top + self.status_height;
        let footer_top = status_bottom + SHARED_CONTROL_ROW_GAP;

        match row {
            ControlRow::Top => ControlRowBounds {
                top: 0.0,
                height: self.top_height,
            },
            ControlRow::QuickPreview => ControlRowBounds {
                top: quick_top,
                height: self.quick_preview_height,
            },
            ControlRow::Dng => ControlRowBounds {
                top: dng_top,
                height: self.dng_height,
            },
            ControlRow::Status => ControlRowBounds {
                top: status_top,
                height: self.status_height,
            },
            ControlRow::Footer => ControlRowBounds {
                top: footer_top,
                height: self.footer_height,
            },
        }
    }

    pub fn row_top(self, row: ControlRow) -> f32 {
        self.row_bounds(row).top
    }

    pub fn row_bottom(self, row: ControlRow) -> f32 {
        self.row_bounds(row).bottom()
    }

    pub fn footer_center(self) -> f32 {
        self.row_bounds(ControlRow::Footer).center()
    }
}

pub fn shared_controls_row_layout(available_height: f32) -> SharedControlsRowLayout {
    let available_height = non_negative_finite(available_height);
    let quick_preview_height = SHARED_QUICK_PREVIEW_ROW_HEIGHT;
    let dng_height = SHARED_DNG_ROW_HEIGHT;
    let status_height = PRIMARY_STATUS_HEIGHT;
    let footer_height = SHARED_FOOTER_ROW_HEIGHT;
    let fixed_lower_control_height = shared_fixed_lower_control_height();
    let minimum_top_height = shared_top_minimum_height();
    let top_height = (available_height - fixed_lower_control_height)
        .max(minimum_top_height)
        .max(0.0);
    let total_used_height = top_height + fixed_lower_control_height;

    SharedControlsRowLayout {
        available_height,
        top_height,
        playlist_list_height: (top_height - PRIMARY_PLAYLIST_FIXED_HEIGHT).max(0.0),
        quick_preview_height,
        dng_height,
        status_height,
        footer_height,
        total_used_height,
        outer_scroll_required: total_used_height > available_height,
    }
}

fn shared_fixed_lower_control_height() -> f32 {
    SHARED_CONTROL_ROW_GAP
        + SHARED_QUICK_PREVIEW_ROW_HEIGHT
        + SHARED_CONTROL_ROW_GAP
        + SHARED_DNG_ROW_HEIGHT
        + SHARED_CONTROL_ROW_GAP
        + PRIMARY_STATUS_HEIGHT
        + SHARED_CONTROL_ROW_GAP
        + SHARED_FOOTER_ROW_HEIGHT
}

fn shared_top_minimum_height() -> f32 {
    let playlist_minimum = PRIMARY_PLAYLIST_FIXED_HEIGHT + PRIMARY_PLAYLIST_MIN_USEFUL_HEIGHT;
    playlist_minimum.max(MIDDLE_TOP_FIXED_HEIGHT)
}

pub fn equal_button_widths(available_width: f32, item_spacing: f32) -> [f32; 2] {
    let width = equal_row_button_width(available_width, item_spacing, 2);
    [width, width]
}

pub fn equal_three_button_widths(available_width: f32, item_spacing: f32) -> [f32; 3] {
    let width = equal_row_button_width(available_width, item_spacing, 3);
    [width, width, width]
}

pub fn dng_grid_columns(available_width: f32, item_spacing: f32) -> [f32; 3] {
    equal_three_button_widths(available_width, item_spacing)
}

fn equal_row_button_width(available_width: f32, item_spacing: f32, button_count: usize) -> f32 {
    if button_count == 0 {
        return 0.0;
    }
    let gaps = button_count.saturating_sub(1) as f32 * item_spacing.max(0.0);
    ((available_width.max(0.0) - gaps) / button_count as f32).max(0.0)
}

pub fn scrubber_width(display_width: f32) -> f32 {
    display_width.max(0.0) * 0.5
}

pub fn playlist_entry_text_tone(state: PlaylistEntryVisualState) -> PlaylistEntryTextTone {
    match state {
        PlaylistEntryVisualState::Normal | PlaylistEntryVisualState::UnmountedByGui => {
            PlaylistEntryTextTone::Gray
        }
        PlaylistEntryVisualState::IntendedMounted => PlaylistEntryTextTone::White,
        PlaylistEntryVisualState::MountedByGui => PlaylistEntryTextTone::BrightBlue,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_condensed_roles_exclude_more_options() {
        assert_eq!(
            MainLayoutMode::Condensed.visible_roles(),
            [
                MainLayoutRole::PrimaryControls,
                MainLayoutRole::DisplayPlayback
            ]
        );
        assert!(
            !MainLayoutMode::Condensed
                .visible_roles()
                .contains(&MainLayoutRole::MoreOptions)
        );
    }

    #[test]
    fn target_expanded_roles_include_more_options_between_primary_and_display() {
        assert_eq!(
            MainLayoutMode::Expanded.visible_roles(),
            [
                MainLayoutRole::PrimaryControls,
                MainLayoutRole::MoreOptions,
                MainLayoutRole::DisplayPlayback
            ]
        );
    }

    #[test]
    fn target_condensed_1600_layout_matches_required_widths() {
        let layout = target_main_layout(MainLayoutMode::Condensed, 1600.0);

        assert_eq!(layout.mode, MainLayoutMode::Condensed);
        assert_eq!(layout.primary_controls_width, 728.0);
        assert_eq!(layout.more_options_width, 0.0);
        assert_eq!(layout.display_width, 808.0);
        assert_eq!(layout.outer_margin, style::OUTER_MARGIN);
        assert_eq!(layout.column_gap, 16.0);
        assert_eq!(layout.total_width(), 1600.0);
        assert!(layout.fits_minimum_display_width(520.0));
    }

    #[test]
    fn target_expanded_1600_layout_matches_required_widths() {
        let layout = target_main_layout(MainLayoutMode::Expanded, 1600.0);

        assert_eq!(layout.mode, MainLayoutMode::Expanded);
        assert_eq!(layout.primary_controls_width, 728.0);
        assert_eq!(layout.more_options_width, 472.0);
        assert_eq!(layout.display_width, 320.0);
        assert_eq!(layout.outer_margin, style::OUTER_MARGIN);
        assert_eq!(layout.column_gap, 16.0);
        assert_eq!(layout.total_width(), 1600.0);
        assert!(!layout.fits_minimum_display_width(520.0));
    }

    #[test]
    fn target_layout_widths_match_1920_reference() {
        let condensed = target_main_layout(MainLayoutMode::Condensed, 1920.0);
        let expanded = target_main_layout(MainLayoutMode::Expanded, 1920.0);

        assert_eq!(condensed.display_width, 1128.0);
        assert_eq!(expanded.display_width, 640.0);
        assert_eq!(condensed.total_width(), 1920.0);
        assert_eq!(expanded.total_width(), 1920.0);
    }

    #[test]
    fn target_layout_widths_match_1366_reference() {
        let condensed = target_main_layout(MainLayoutMode::Condensed, 1366.0);
        let expanded = target_main_layout(MainLayoutMode::Expanded, 1366.0);

        assert_eq!(condensed.display_width, 574.0);
        assert_eq!(expanded.display_width, 86.0);
        assert_eq!(condensed.total_width(), 1366.0);
        assert_eq!(expanded.total_width(), 1366.0);
        assert!(condensed.fits_minimum_display_width(520.0));
        assert!(!expanded.fits_minimum_display_width(520.0));
    }

    #[test]
    fn target_minimum_window_widths_include_fixed_regions_gaps_and_margins() {
        assert_eq!(
            minimum_window_width(MainLayoutMode::Condensed, 520.0),
            1312.0
        );
        assert_eq!(
            minimum_window_width(MainLayoutMode::Expanded, 520.0),
            1800.0
        );

        let expanded = target_main_layout(MainLayoutMode::Expanded, 1600.0);
        assert_eq!(expanded.minimum_required_width(520.0), 1800.0);
    }

    #[test]
    fn target_layout_clamps_too_small_or_invalid_widths_without_nan() {
        for available_width in [-1.0, 0.0, 100.0, f32::NAN, f32::INFINITY] {
            for mode in [MainLayoutMode::Condensed, MainLayoutMode::Expanded] {
                let layout = target_main_layout(mode, available_width);

                assert_eq!(layout.primary_controls_width, 728.0);
                assert!(layout.more_options_width >= 0.0);
                assert_eq!(layout.display_width, 0.0);
                assert!(!layout.display_width.is_nan());
                assert!(!layout.total_width().is_nan());
                assert!(!layout.fits_minimum_display_width(520.0));
            }
        }
    }

    #[test]
    fn target_display_consumes_all_remaining_sufficient_width() {
        let layout = target_main_layout(MainLayoutMode::Expanded, 2200.0);
        let expected_display = 2200.0
            - style::OUTER_MARGIN * 2.0
            - PRIMARY_CONTROLS_WIDTH
            - MORE_OPTIONS_WIDTH
            - MAIN_COLUMN_GAP * 2.0;

        assert_eq!(layout.display_width, expected_display);
        assert_eq!(layout.total_width(), 2200.0);
    }

    #[test]
    fn column_content_height_removes_frame_margins() {
        assert_eq!(column_content_height(920.0), 884.0);
    }

    #[test]
    fn primary_controls_width_is_derived_from_wider_three_button_rows() {
        let derived = derived_primary_controls_width(STANDARD_ITEM_SPACING);

        assert_eq!(BASE_PRIMARY_CONTROLS_WIDTH, 560.0);
        assert_eq!(PRIMARY_CONTROLS_WIDTH, 728.0);
        assert!(PRIMARY_CONTROLS_WIDTH + 0.01 >= derived);
        assert!((PRIMARY_CONTROLS_WIDTH - derived).abs() < 0.01);
        assert_eq!(
            ordinary_primary_three_button_width(STANDARD_ITEM_SPACING)
                - base_primary_three_button_width(STANDARD_ITEM_SPACING),
            BUTTON_THREE_CHARACTER_EXTRA_WIDTH
        );
    }

    #[test]
    fn more_options_width_is_derived_from_wider_two_button_rows() {
        assert_eq!(BASE_MORE_OPTIONS_WIDTH, 360.0);
        assert_eq!(MORE_OPTIONS_WIDTH, 472.0);
        assert_eq!(
            derived_more_options_width(STANDARD_ITEM_SPACING),
            MORE_OPTIONS_WIDTH
        );
        assert_eq!(
            ordinary_more_options_two_button_width(STANDARD_ITEM_SPACING)
                - base_more_options_two_button_width(STANDARD_ITEM_SPACING),
            BUTTON_THREE_CHARACTER_EXTRA_WIDTH
        );
    }

    #[test]
    fn equal_button_widths_match_for_paired_options() {
        assert_eq!(equal_button_widths(436.0, 8.0), [214.0, 214.0]);
    }

    #[test]
    fn equal_three_button_widths_match_primary_controls_policy() {
        let widths = equal_three_button_widths(
            PRIMARY_CONTROLS_WIDTH - COLUMN_FRAME_INNER_MARGIN * 2.0,
            8.0,
        );

        for width in widths {
            assert!((width - 225.33333).abs() < 0.01);
        }
    }

    #[test]
    fn dng_grid_columns_are_equal_button_columns() {
        let columns = dng_grid_columns(
            PRIMARY_CONTROLS_WIDTH - COLUMN_FRAME_INNER_MARGIN * 2.0,
            8.0,
        );

        assert_eq!(columns[0], columns[1]);
        assert_eq!(columns[1], columns[2]);
        assert_eq!(columns, equal_three_button_widths(692.0, 8.0));
    }

    #[test]
    fn primary_column_budget_fits_corrected_rows_without_outer_scroll() {
        let budget = primary_column_height_budget(792.0);

        assert_eq!(budget.available_height, 792.0);
        assert_eq!(budget.fixed_lower_control_height, 570.0);
        assert_eq!(budget.playlist_fixed_height, 95.0);
        assert_eq!(budget.playlist_list_height, 127.0);
        assert_eq!(budget.total_used_height, 792.0);
        assert!(!budget.outer_scroll_required);
        assert!(budget.dng_bottom() <= budget.available_height);
        assert!(budget.status_bottom() <= budget.available_height);
        assert!(budget.footer_bottom() <= budget.available_height);
    }

    #[test]
    fn primary_column_playlist_list_height_is_useful_at_default() {
        let budget = primary_column_height_budget(792.0);

        assert!(budget.playlist_list_height >= PRIMARY_PLAYLIST_MIN_USEFUL_HEIGHT);
    }

    #[test]
    fn quick_preview_row_minimum_uses_larger_middle_content_requirement() {
        assert_eq!(UNDERSIZED_QUICK_PREVIEW_ROW_HEIGHT, 110.0);
        assert_eq!(QUICK_PREVIEW_LEFT_CONTENT_HEIGHT, 98.0);
        assert_eq!(QUICK_PREVIEW_MIDDLE_CONTENT_HEIGHT, 126.0);
        assert_eq!(
            QUICK_PREVIEW_CONTENT_HEIGHT,
            QUICK_PREVIEW_MIDDLE_CONTENT_HEIGHT
        );
        assert_eq!(CONTROL_ROW_TOP_PADDING, 8.0);
        assert_eq!(CONTROL_ROW_BOTTOM_PADDING, 8.0);
        assert_eq!(SHARED_QUICK_PREVIEW_ROW_HEIGHT, 142.0);
    }

    #[test]
    fn dng_row_minimum_includes_two_button_rows_and_padding() {
        assert_eq!(UNDERSIZED_DNG_ROW_HEIGHT, 166.0);
        assert_eq!(DNG_CONTENT_HEIGHT, 190.0);
        assert_eq!(CONTROL_ROW_TOP_PADDING, 8.0);
        assert_eq!(CONTROL_ROW_BOTTOM_PADDING, 8.0);
        assert_eq!(SHARED_DNG_ROW_HEIGHT, 206.0);
    }

    #[test]
    fn footer_row_minimum_includes_button_top_padding_and_bottom_inset() {
        assert_eq!(UNDERSIZED_FOOTER_ROW_HEIGHT, 34.0);
        assert_eq!(FOOTER_ROW_TOP_PADDING, 8.0);
        assert_eq!(FOOTER_BOTTOM_INSET, 8.0);
        assert_eq!(SHARED_FOOTER_ROW_HEIGHT, 50.0);
    }

    #[test]
    fn primary_column_budget_grows_playlist_on_taller_windows() {
        let default = primary_column_height_budget(792.0);
        let taller = primary_column_height_budget(900.0);

        assert_eq!(
            default.fixed_lower_control_height,
            taller.fixed_lower_control_height
        );
        assert_eq!(default.playlist_fixed_height, taller.playlist_fixed_height);
        assert!(taller.playlist_list_height > default.playlist_list_height);
        assert!(!taller.outer_scroll_required);
    }

    #[test]
    fn primary_column_budget_uses_scroll_fallback_when_smaller_height_cannot_fit() {
        let budget = primary_column_height_budget(768.0);

        assert!(budget.outer_scroll_required);
        assert!(budget.playlist_list_height >= PRIMARY_PLAYLIST_MIN_USEFUL_HEIGHT);
        assert!(budget.total_used_height > budget.available_height);
    }

    #[test]
    fn shared_row_layout_aligns_left_and_middle_row_boundaries() {
        let rows = shared_controls_row_layout(792.0);

        assert_eq!(rows.row_top(ControlRow::Top), 0.0);
        assert_eq!(
            rows.row_top(ControlRow::QuickPreview),
            rows.row_bottom(ControlRow::Top) + SHARED_CONTROL_ROW_GAP
        );
        assert_eq!(
            rows.row_top(ControlRow::Dng),
            rows.row_bottom(ControlRow::QuickPreview) + SHARED_CONTROL_ROW_GAP
        );
        assert_eq!(
            rows.row_top(ControlRow::Status),
            rows.row_bottom(ControlRow::Dng) + SHARED_CONTROL_ROW_GAP
        );
        assert_eq!(
            rows.row_top(ControlRow::Footer),
            rows.row_bottom(ControlRow::Status) + SHARED_CONTROL_ROW_GAP
        );
        assert_eq!(rows.row_bottom(ControlRow::Footer), rows.total_used_height);
    }

    #[test]
    fn shared_row_layout_keeps_footer_visible_at_default_and_1440_reference() {
        let default = shared_controls_row_layout(792.0);
        let reference_1440 = shared_controls_row_layout(1260.0);

        assert!(!default.outer_scroll_required);
        assert!(default.row_bottom(ControlRow::Footer) <= default.available_height);
        assert!(!reference_1440.outer_scroll_required);
        assert!(reference_1440.row_bottom(ControlRow::Footer) <= reference_1440.available_height);
        assert!(reference_1440.playlist_list_height > default.playlist_list_height);
    }

    #[test]
    fn shared_row_layout_uses_top_playlist_as_flexible_row() {
        let default = shared_controls_row_layout(792.0);
        let taller = shared_controls_row_layout(900.0);

        assert_eq!(default.quick_preview_height, taller.quick_preview_height);
        assert_eq!(default.dng_height, taller.dng_height);
        assert_eq!(default.status_height, taller.status_height);
        assert_eq!(default.footer_height, taller.footer_height);
        assert!(taller.top_height > default.top_height);
        assert!(taller.playlist_list_height > default.playlist_list_height);
    }

    #[test]
    fn shared_row_layout_small_height_requests_one_outer_scroll_fallback() {
        let rows = shared_controls_row_layout(768.0);

        assert!(rows.outer_scroll_required);
        assert!(rows.total_used_height > rows.available_height);
        assert!(rows.top_height >= MIDDLE_TOP_FIXED_HEIGHT);
    }

    #[test]
    fn shared_row_layout_clamps_without_negative_or_nan_dimensions() {
        for available_height in [-1.0, 0.0, 100.0, f32::NAN, f32::INFINITY] {
            let rows = shared_controls_row_layout(available_height);

            for row in [
                ControlRow::Top,
                ControlRow::QuickPreview,
                ControlRow::Dng,
                ControlRow::Status,
                ControlRow::Footer,
            ] {
                let bounds = rows.row_bounds(row);
                assert!(bounds.top >= 0.0);
                assert!(bounds.height >= 0.0);
                assert!(!bounds.top.is_nan());
                assert!(!bounds.height.is_nan());
                assert!(!bounds.bottom().is_nan());
            }
            assert!(rows.playlist_list_height >= 0.0);
            assert!(!rows.playlist_list_height.is_nan());
            assert!(!rows.total_used_height.is_nan());
        }
    }

    #[test]
    fn primary_column_budget_clamps_invalid_or_tiny_heights_without_nan() {
        for available_height in [-1.0, 0.0, 100.0, f32::NAN, f32::INFINITY] {
            let budget = primary_column_height_budget(available_height);

            assert!(budget.available_height >= 0.0);
            assert!(budget.fixed_lower_control_height >= 0.0);
            assert!(budget.playlist_fixed_height >= 0.0);
            assert!(budget.playlist_list_height >= 0.0);
            assert!(budget.total_used_height >= 0.0);
            assert!(!budget.available_height.is_nan());
            assert!(!budget.total_used_height.is_nan());
        }
    }

    #[test]
    fn scrubber_width_is_half_display_width() {
        assert_eq!(scrubber_width(800.0), 400.0);
        assert_eq!(scrubber_width(1920.0), 960.0);
    }

    #[test]
    fn playlist_entry_visual_policy_maps_to_text_tones() {
        assert_eq!(
            playlist_entry_text_tone(PlaylistEntryVisualState::Normal),
            PlaylistEntryTextTone::Gray
        );
        assert_eq!(
            playlist_entry_text_tone(PlaylistEntryVisualState::IntendedMounted),
            PlaylistEntryTextTone::White
        );
        assert_eq!(
            playlist_entry_text_tone(PlaylistEntryVisualState::MountedByGui),
            PlaylistEntryTextTone::BrightBlue
        );
        assert_eq!(
            playlist_entry_text_tone(PlaylistEntryVisualState::UnmountedByGui),
            PlaylistEntryTextTone::Gray
        );
    }

    #[test]
    fn skeleton_labels_include_expected_sections() {
        for label in [
            "No clip playing",
            "Primary Controls",
            "Quick Preview",
            "Decoding Settings",
            "Optimized Decode Settings",
            "Compares Default and Offset payload profiles for this system",
            "Vsync On",
            "Max speed",
            "FPS overlay",
            "Add files",
            "Remove file",
            "Remove all",
            "Mount all DNGs",
            "Pipe Example",
            "Playlist",
            "No files in playlist.",
        ] {
            assert!(MAIN_SKELETON_LABELS.contains(&label));
        }

        for old_label in [
            "Preview with Vsync",
            "Preview at max decode speed",
            "FPS graphical overlay",
            "Add file to playlist",
            "Remove from Playlist",
            "DNG Decode Settings",
        ] {
            assert!(!MAIN_SKELETON_LABELS.contains(&old_label));
        }
    }
}

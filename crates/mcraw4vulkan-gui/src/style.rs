use egui::{Color32, FontFamily, FontId, Stroke, Style, TextStyle, Vec2, Visuals, pos2, vec2};

pub const BACKGROUND_RGB: [u8; 3] = [18, 18, 18];
pub const PANEL_RGB: [u8; 3] = [24, 24, 24];
pub const HEADER_TEXT_RGB: [u8; 3] = [245, 245, 245];
pub const BODY_TEXT_RGB: [u8; 3] = [180, 180, 180];
pub const MUTED_TEXT_RGB: [u8; 3] = [125, 125, 125];
pub const DIVIDER_RGB: [u8; 3] = [70, 70, 70];
pub const BRIGHT_BLUE_RGB: [u8; 3] = [0, 145, 210];
pub const BUTTON_DARK_RGB: [u8; 3] = [55, 55, 55];
pub const ORANGE_EMPHASIS_RGB: [u8; 3] = [206, 123, 91];

pub const OUTER_MARGIN: f32 = 24.0;
pub const HEADER_HEIGHT: f32 = 72.0;
pub const SPLASH_PANEL_WIDTH: f32 = 800.0;
pub const HEADING_FONT_SIZE: f32 = 30.0;
pub const BODY_FONT_SIZE: f32 = 20.0;
pub const BUTTON_FONT_SIZE: f32 = 20.0;
pub const SMALL_FONT_SIZE: f32 = 18.0;
pub const PANEL_TITLE_FONT_SIZE: f32 = 28.0;
pub const STATUS_FONT_SIZE: f32 = 22.0;
pub const DETAIL_FONT_SIZE: f32 = 20.0;
pub const LINE_FONT_SIZE: f32 = 20.0;

pub fn background() -> Color32 {
    rgb(BACKGROUND_RGB)
}

pub fn panel() -> Color32 {
    rgb(PANEL_RGB)
}

pub fn header_text() -> Color32 {
    rgb(HEADER_TEXT_RGB)
}

pub fn body_text() -> Color32 {
    rgb(BODY_TEXT_RGB)
}

pub fn muted_text() -> Color32 {
    rgb(MUTED_TEXT_RGB)
}

pub fn divider() -> Color32 {
    rgb(DIVIDER_RGB)
}

pub fn bright_blue() -> Color32 {
    rgb(BRIGHT_BLUE_RGB)
}

pub fn button_dark() -> Color32 {
    rgb(BUTTON_DARK_RGB)
}

pub fn orange_emphasis() -> Color32 {
    rgb(ORANGE_EMPHASIS_RGB)
}

pub fn rgb(value: [u8; 3]) -> Color32 {
    Color32::from_rgb(value[0], value[1], value[2])
}

pub fn with_orange_emphasis_button_style<R>(
    ui: &mut egui::Ui,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    ui.scope(|ui| {
        apply_orange_emphasis_button_visuals(&mut ui.style_mut().visuals);
        add_contents(ui)
    })
    .inner
}

pub fn apply_orange_emphasis_button_visuals(visuals: &mut Visuals) {
    let stroke_color = orange_emphasis();
    visuals.widgets.noninteractive.bg_stroke.color = stroke_color;
    visuals.widgets.inactive.bg_stroke.color = stroke_color;
    visuals.widgets.hovered.bg_stroke.color = stroke_color;
    visuals.widgets.active.bg_stroke.color = stroke_color;
    visuals.widgets.open.bg_stroke.color = stroke_color;
}

pub fn apply_project_style(context: &egui::Context) {
    let mut style = Style::default();
    style.text_styles.insert(
        TextStyle::Heading,
        FontId::new(HEADING_FONT_SIZE, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Body,
        FontId::new(BODY_FONT_SIZE, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Button,
        FontId::new(BUTTON_FONT_SIZE, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Small,
        FontId::new(SMALL_FONT_SIZE, FontFamily::Proportional),
    );
    style.spacing.item_spacing = vec2(8.0, 8.0);
    style.spacing.button_padding = vec2(12.0, 6.0);
    style.spacing.interact_size = vec2(56.0, 32.0);
    style.visuals = project_visuals();
    context.set_style(style);
}

pub fn project_visuals() -> Visuals {
    let mut visuals = Visuals::dark();
    visuals.override_text_color = Some(body_text());
    visuals.panel_fill = background();
    visuals.window_fill = panel();
    visuals.window_stroke = Stroke::new(1.0, divider());
    visuals.faint_bg_color = panel();
    visuals.extreme_bg_color = background();
    visuals.hyperlink_color = bright_blue();
    visuals.selection.bg_fill = bright_blue();
    visuals.selection.stroke = Stroke::new(1.0, header_text());
    visuals.warn_fg_color = body_text();
    visuals.error_fg_color = body_text();

    visuals.widgets.noninteractive.bg_fill = panel();
    visuals.widgets.noninteractive.weak_bg_fill = panel();
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, divider());
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, body_text());

    visuals.widgets.inactive.bg_fill = button_dark();
    visuals.widgets.inactive.weak_bg_fill = button_dark();
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, divider());
    visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, body_text());

    visuals.widgets.hovered.bg_fill = bright_blue();
    visuals.widgets.hovered.weak_bg_fill = bright_blue();
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, bright_blue());
    visuals.widgets.hovered.fg_stroke = Stroke::new(1.0, header_text());

    visuals.widgets.active.bg_fill = bright_blue();
    visuals.widgets.active.weak_bg_fill = bright_blue();
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, bright_blue());
    visuals.widgets.active.fg_stroke = Stroke::new(1.0, header_text());
    visuals
}

pub fn logical_screen_rect(width: u32, height: u32) -> egui::Rect {
    egui::Rect::from_min_size(pos2(0.0, 0.0), Vec2::new(width as f32, height as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_constants_match_project_palette() {
        assert_eq!(background(), Color32::from_rgb(18, 18, 18));
        assert_eq!(panel(), Color32::from_rgb(24, 24, 24));
        assert_eq!(header_text(), Color32::from_rgb(245, 245, 245));
        assert_eq!(body_text(), Color32::from_rgb(180, 180, 180));
        assert_eq!(muted_text(), Color32::from_rgb(125, 125, 125));
        assert_eq!(divider(), Color32::from_rgb(70, 70, 70));
        assert_eq!(bright_blue(), Color32::from_rgb(0, 145, 210));
        assert_eq!(button_dark(), Color32::from_rgb(55, 55, 55));
    }

    #[test]
    fn orange_emphasis_matches_e34_color_contract() {
        assert_eq!(ORANGE_EMPHASIS_RGB, [206, 123, 91]);
        assert_eq!(orange_emphasis(), Color32::from_rgb(206, 123, 91));
        assert_eq!(orange_emphasis().a(), 255);
    }

    #[test]
    fn project_visuals_forces_dark_theme_and_blue_selection() {
        let visuals = project_visuals();

        assert!(visuals.dark_mode);
        assert_eq!(visuals.selection.bg_fill, bright_blue());
        assert_eq!(visuals.panel_fill, background());
    }

    #[test]
    fn project_visuals_keep_ordinary_button_palette() {
        let visuals = project_visuals();

        assert_eq!(visuals.widgets.inactive.bg_fill, button_dark());
        assert_eq!(visuals.widgets.inactive.weak_bg_fill, button_dark());
        assert_eq!(
            visuals.widgets.inactive.bg_stroke,
            Stroke::new(1.0, divider())
        );
        assert_eq!(
            visuals.widgets.inactive.fg_stroke,
            Stroke::new(1.0, body_text())
        );
        assert_eq!(visuals.widgets.hovered.bg_fill, bright_blue());
        assert_eq!(visuals.widgets.hovered.weak_bg_fill, bright_blue());
        assert_eq!(
            visuals.widgets.hovered.bg_stroke,
            Stroke::new(1.0, bright_blue())
        );
        assert_eq!(
            visuals.widgets.hovered.fg_stroke,
            Stroke::new(1.0, header_text())
        );
        assert_eq!(visuals.widgets.active.bg_fill, bright_blue());
        assert_eq!(visuals.widgets.active.weak_bg_fill, bright_blue());
        assert_eq!(
            visuals.widgets.active.bg_stroke,
            Stroke::new(1.0, bright_blue())
        );
        assert_eq!(
            visuals.widgets.active.fg_stroke,
            Stroke::new(1.0, header_text())
        );
    }

    #[test]
    fn orange_emphasis_button_visuals_only_recolor_widget_outlines() {
        let mut visuals = project_visuals();
        let before = visuals.clone();

        apply_orange_emphasis_button_visuals(&mut visuals);

        assert_orange_emphasis_preserves_state(
            before.widgets.noninteractive,
            visuals.widgets.noninteractive,
        );
        assert_orange_emphasis_preserves_state(before.widgets.inactive, visuals.widgets.inactive);
        assert_orange_emphasis_preserves_state(before.widgets.hovered, visuals.widgets.hovered);
        assert_orange_emphasis_preserves_state(before.widgets.active, visuals.widgets.active);
        assert_orange_emphasis_preserves_state(before.widgets.open, visuals.widgets.open);
    }

    #[test]
    fn project_visuals_remain_ordinary_without_orange_emphasis_scope() {
        let before = project_visuals();
        let mut scoped = before.clone();

        apply_orange_emphasis_button_visuals(&mut scoped);
        let ordinary = project_visuals();

        assert_eq!(ordinary.widgets.inactive.bg_fill, button_dark());
        assert_eq!(ordinary.widgets.hovered.bg_fill, bright_blue());
        assert_eq!(ordinary.widgets.active.bg_fill, bright_blue());
        assert_eq!(
            ordinary.widgets.inactive.bg_stroke,
            Stroke::new(1.0, divider())
        );
        assert_eq!(ordinary, before);
        assert_eq!(scoped.widgets.inactive.bg_stroke.color, orange_emphasis());
    }

    fn assert_orange_emphasis_preserves_state(
        before: egui::style::WidgetVisuals,
        after: egui::style::WidgetVisuals,
    ) {
        assert_eq!(after.bg_stroke.color, orange_emphasis());
        assert_eq!(after.bg_stroke.width, before.bg_stroke.width);
        assert_eq!(after.bg_fill, before.bg_fill);
        assert_eq!(after.weak_bg_fill, before.weak_bg_fill);
        assert_eq!(after.fg_stroke, before.fg_stroke);
        assert_eq!(after.expansion, before.expansion);
        assert_eq!(after.corner_radius, before.corner_radius);
    }

    #[test]
    fn project_style_uses_b08_font_sizes() {
        let context = egui::Context::default();

        apply_project_style(&context);
        let style = context.style();

        assert_eq!(
            style.text_styles.get(&TextStyle::Heading).unwrap().size,
            30.0
        );
        assert_eq!(style.text_styles.get(&TextStyle::Body).unwrap().size, 20.0);
        assert_eq!(
            style.text_styles.get(&TextStyle::Button).unwrap().size,
            20.0
        );
        assert_eq!(style.text_styles.get(&TextStyle::Small).unwrap().size, 18.0);
    }

    #[test]
    fn font_size_constants_are_even() {
        for size in [
            HEADING_FONT_SIZE,
            BODY_FONT_SIZE,
            BUTTON_FONT_SIZE,
            SMALL_FONT_SIZE,
            PANEL_TITLE_FONT_SIZE,
            STATUS_FONT_SIZE,
            DETAIL_FONT_SIZE,
            LINE_FONT_SIZE,
        ] {
            assert_eq!(size % 2.0, 0.0);
        }
    }

    #[test]
    fn logical_rect_uses_window_points() {
        let rect = logical_screen_rect(1600, 1080);

        assert_eq!(rect.width(), 1600.0);
        assert_eq!(rect.height(), 1080.0);
    }
}

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

/// Density is selected from the full client in logical points, before local layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Density {
    #[default]
    Normal,
    Compact,
}

impl Density {
    pub fn for_client(size: Vec2) -> Self {
        if size.x >= 1664.0 && size.y >= 1012.0 {
            Self::Normal
        } else {
            Self::Compact
        }
    }
    pub fn font(self, normal: f32) -> f32 {
        normal * if self == Self::Compact { 0.9 } else { 1.0 }
    }
    pub fn margin(self) -> f32 {
        if self == Self::Compact { 4.0 } else { 24.0 }
    }
    pub fn inset(self) -> f32 {
        if self == Self::Compact { 4.0 } else { 18.0 }
    }
    pub fn gutter(self) -> f32 {
        if self == Self::Compact { 4.0 } else { 16.0 }
    }
    pub fn item_gap(self) -> f32 {
        if self == Self::Compact { 4.0 } else { 8.0 }
    }
    pub fn row_padding(self) -> f32 {
        if self == Self::Compact { 0.0 } else { 8.0 }
    }
    pub fn list_inset(self) -> f32 {
        if self == Self::Compact { 8.0 } else { 16.0 }
    }
    pub fn body_inset(self) -> f32 {
        if self == Self::Compact { 12.0 } else { 24.0 }
    }
    pub fn space(self, normal: f32) -> f32 {
        if self == Self::Normal {
            return normal;
        }
        match normal {
            18.0 | 16.0 | 14.0 | 12.0 | 8.0 => 4.0,
            6.0 => 3.0,
            _ => normal * 0.9,
        }
    }
}

fn density_id() -> egui::Id {
    egui::Id::new("layout-density")
}
pub fn density(context: &egui::Context) -> Density {
    context.data(|data| data.get_temp::<Density>(density_id()).unwrap_or_default())
}
pub fn metrics(ui: &egui::Ui) -> Density {
    density(ui.ctx())
}

pub fn select_density(context: &egui::Context) {
    let next = Density::for_client(context.input(|input| input.screen_rect().size()));
    if next != density(context) {
        apply_density(context, next);
    }
}

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
    apply_density(context, Density::Normal);
}

pub fn apply_density(context: &egui::Context, density: Density) {
    let mut style = Style::default();
    style.text_styles.insert(
        TextStyle::Heading,
        FontId::new(density.font(HEADING_FONT_SIZE), FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Body,
        FontId::new(density.font(BODY_FONT_SIZE), FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Button,
        FontId::new(density.font(BUTTON_FONT_SIZE), FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Small,
        FontId::new(density.font(SMALL_FONT_SIZE), FontFamily::Proportional),
    );
    style.spacing.item_spacing = Vec2::splat(density.item_gap());
    style.spacing.button_padding = if density == Density::Compact {
        vec2(6.0, 4.0)
    } else {
        vec2(12.0, 6.0)
    };
    style.spacing.interact_size = if density == Density::Compact {
        vec2(50.4, 30.6)
    } else {
        vec2(56.0, 32.0)
    };
    style.visuals = project_visuals();
    context.set_style(style);
    context.data_mut(|data| data.insert_temp(density_id(), density));
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

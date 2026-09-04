use mcraw4vulkan_render::SIGNED_BAYER_DEMOSAIC_WGSL;

#[test]
fn signed_fragment_declares_no_clamp_or_second_correction() {
    for forbidden in [
        "clamp(",
        "max(",
        "min(",
        "black_for_",
        "white_level",
        "gain_q",
        "corrected_code",
        "u16(",
    ] {
        assert!(
            !SIGNED_BAYER_DEMOSAIC_WGSL.contains(forbidden),
            "signed fragment unexpectedly contains {forbidden:?}"
        );
    }
    assert!(SIGNED_BAYER_DEMOSAIC_WGSL.contains("load_pipe_f32_bayer_index"));
    assert!(SIGNED_BAYER_DEMOSAIC_WGSL.contains("demosaic_signed_bayer"));
}

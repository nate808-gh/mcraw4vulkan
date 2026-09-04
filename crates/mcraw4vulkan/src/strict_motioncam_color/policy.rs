pub(super) const POLICY_MAGIC: &[u8; 56] =
    b"mcraw4vulkan:StrictMotionCamForwardMatrixColorV2:policy\0";
pub(super) const CONTEXT_MAGIC: &[u8; 57] =
    b"mcraw4vulkan:StrictMotionCamForwardMatrixColorV2:context\0";
pub(super) const POLICY_SCHEMA: u32 = 2;
pub(super) const CONTEXT_SCHEMA: u32 = 2;
pub(super) const POLICY_RECORD_BYTE_LEN: usize = 1_404;
pub(super) const EXPECTED_POLICY_DIGEST: [u8; 32] = [
    0xd6, 0xad, 0x75, 0x70, 0xd1, 0xd8, 0x43, 0xbd, 0xef, 0xbb, 0x20, 0xa6, 0x26, 0x0e, 0x86, 0x7f,
    0x62, 0xe1, 0x50, 0x96, 0xb2, 0xd3, 0xf5, 0xa7, 0xf1, 0x39, 0x1d, 0x69, 0x08, 0x3e, 0x37, 0x1e,
];

pub(super) const D50_XY: [f64; 2] = [0.3457, 0.3585];
pub(super) const D65_XY: [f64; 2] = [0.3127, 0.3290];
pub(super) const STANDARD_A_XY: [f64; 2] = [0.4476, 0.4074];
pub(super) const STANDARD_A_TEMPERATURE: f64 = 2850.0;
pub(super) const D65_TEMPERATURE: f64 = 6500.0;
pub(super) const WHITE_TOLERANCE_L1_XY: f64 = 1.0e-7;
pub(super) const WHITE_MAX_ITERATIONS: u32 = 64;
pub(super) const ASN_NORMALIZATION_MAX: f64 = 1.0;
pub(super) const MAX_CONDITION_1: f64 = 1.0e3;
pub(super) const MIN_RELATIVE_DETERMINANT: f64 = 1.0e-12;
pub(super) const MAX_INPUT_ELEMENT: f64 = 64.0;
pub(super) const MAX_COMPOSITE_ELEMENT: f64 = 256.0;
pub(super) const MIN_FORWARD_NORMALIZATION_ROW_SUM: f64 = 1.0e-12;
pub(super) const WGSL_FINITE_SAFETY_DIVISOR: f64 = 2.0;

pub(super) const BRADFORD_CONE: [f64; 9] = [
    0.8951, 0.2664, -0.1614, -0.7502, 1.7135, 0.0367, 0.0389, -0.0685, 1.0296,
];

pub(super) const BRADFORD_D50_TO_D65: [f64; 9] = [
    0.9554734214880751,
    -0.023098454948764627,
    0.06325924320057069,
    -0.0283697093338637,
    1.0099953980813043,
    0.021041441191917306,
    0.01231401486448199,
    -0.020507649298898974,
    1.3303659262421241,
];

pub(super) const XYZ_D65_TO_LINEAR_BT2020: [f64; 9] = [
    1.716651187971268,
    -0.355670783776392,
    -0.25336628137366,
    -0.666684351832489,
    1.616481236634939,
    0.015768545813911,
    0.017639857445311,
    -0.042770613257809,
    0.942103121235474,
];

pub(super) const LINEAR_BT2020_TO_NORMALIZED_NCL: [f64; 9] = [
    0.2627,
    0.6780,
    0.0593,
    -0.13963006271925163,
    -0.36036993728074837,
    0.5,
    0.5,
    -0.45978570459785704,
    -0.04021429540214296,
];

/// Adobe DNG SDK dng_temperature.cpp kTempTable, retained in exact order.
pub(super) const TEMPERATURE_TABLE: [[f64; 4]; 31] = [
    [0.0, 0.18006, 0.26352, -0.24341],
    [10.0, 0.18066, 0.26589, -0.25479],
    [20.0, 0.18133, 0.26846, -0.26876],
    [30.0, 0.18208, 0.27119, -0.28539],
    [40.0, 0.18293, 0.27407, -0.30470],
    [50.0, 0.18388, 0.27709, -0.32675],
    [60.0, 0.18494, 0.28021, -0.35156],
    [70.0, 0.18611, 0.28342, -0.37915],
    [80.0, 0.18740, 0.28668, -0.40955],
    [90.0, 0.18880, 0.28997, -0.44278],
    [100.0, 0.19032, 0.29326, -0.47888],
    [125.0, 0.19462, 0.30141, -0.58204],
    [150.0, 0.19962, 0.30921, -0.70471],
    [175.0, 0.20525, 0.31647, -0.84901],
    [200.0, 0.21142, 0.32312, -1.0182],
    [225.0, 0.21807, 0.32909, -1.2168],
    [250.0, 0.22511, 0.33439, -1.4512],
    [275.0, 0.23247, 0.33904, -1.7298],
    [300.0, 0.24010, 0.34308, -2.0637],
    [325.0, 0.24702, 0.34655, -2.4681],
    [350.0, 0.25591, 0.34951, -2.9641],
    [375.0, 0.26400, 0.35200, -3.5814],
    [400.0, 0.27218, 0.35407, -4.3633],
    [425.0, 0.28039, 0.35577, -5.3762],
    [450.0, 0.28863, 0.35714, -6.7262],
    [475.0, 0.29685, 0.35823, -8.5955],
    [500.0, 0.30505, 0.35907, -11.324],
    [525.0, 0.31320, 0.35968, -15.628],
    [550.0, 0.32129, 0.36011, -23.325],
    [575.0, 0.32931, 0.36038, -40.770],
    [600.0, 0.33724, 0.36051, -116.45],
];

pub(super) fn policy_record() -> Vec<u8> {
    let mut output = Vec::with_capacity(POLICY_RECORD_BYTE_LEN);
    output.extend_from_slice(POLICY_MAGIC);
    push_u32(&mut output, POLICY_SCHEMA);
    for value in [
        D50_XY[0],
        D50_XY[1],
        D65_XY[0],
        D65_XY[1],
        STANDARD_A_XY[0],
        STANDARD_A_XY[1],
        STANDARD_A_TEMPERATURE,
        D65_TEMPERATURE,
        WHITE_TOLERANCE_L1_XY,
        ASN_NORMALIZATION_MAX,
    ] {
        push_f64(&mut output, value);
    }
    push_u32(&mut output, WHITE_MAX_ITERATIONS);
    for value in [
        MAX_CONDITION_1,
        MIN_RELATIVE_DETERMINANT,
        MAX_INPUT_ELEMENT,
        MAX_COMPOSITE_ELEMENT,
        MIN_FORWARD_NORMALIZATION_ROW_SUM,
        WGSL_FINITE_SAFETY_DIVISOR,
    ] {
        push_f64(&mut output, value);
    }
    for value in BRADFORD_CONE
        .into_iter()
        .chain(XYZ_D65_TO_LINEAR_BT2020)
        .chain(LINEAR_BT2020_TO_NORMALIZED_NCL)
    {
        push_f64(&mut output, value);
    }
    push_u32(
        &mut output,
        u32::try_from(TEMPERATURE_TABLE.len()).expect("temperature table length fits u32"),
    );
    for row in TEMPERATURE_TABLE {
        for value in row {
            push_f64(&mut output, value);
        }
    }
    output
}

pub(super) fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

pub(super) fn push_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

pub(super) fn push_i64(output: &mut Vec<u8>, value: i64) {
    output.extend_from_slice(&value.to_le_bytes());
}

pub(super) fn push_f64(output: &mut Vec<u8>, value: f64) {
    output.extend_from_slice(&value.to_bits().to_le_bytes());
}

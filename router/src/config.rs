use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "model_type")]
#[serde(rename_all = "snake_case")]
pub struct LlavaNext {
    pub(crate) text_config: TextConfig,
    pub(crate) vision_config: VisionConfig,
    pub(crate) image_grid_pinpoints: Vec<(usize, usize)>,
}

fn get_anyres_image_grid_shape(
    height: usize,
    width: usize,
    grid_pinpoints: &[(usize, usize)],
    patch_size: usize,
) -> (usize, usize) {
    let (height, width) = select_best_resolution(height, width, grid_pinpoints);
    (height / patch_size, width / patch_size)
}

/// Selects the best resolution from a list of possible resolutions based on the original size.
/// This is done by calculating the effective and wasted resolution for each possible resolution.
/// The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.
fn select_best_resolution(
    original_height: usize,
    original_width: usize,
    possible_resolutions: &[(usize, usize)],
) -> (usize, usize) {
    let mut best_fit = None;
    let mut max_effective_resolution = 0;
    let mut min_wasted_resolution = f32::NEG_INFINITY;

    for (height, width) in possible_resolutions {
        let wscale = *width as f32 / original_width as f32;
        let hscale = *height as f32 / original_height as f32;
        // f32 partial ord.
        let scale = if wscale > hscale { hscale } else { wscale };
        let downscaled_width = (*width as f32 * scale) as usize;
        let downscaled_height = (*height as f32 * scale) as usize;
        let effective_resolution = std::cmp::min(
            downscaled_width * downscaled_height,
            original_width * original_height,
        );
        let wasted_resolution = (width * height) - effective_resolution;

        if effective_resolution > max_effective_resolution
            || (effective_resolution == max_effective_resolution
                && (wasted_resolution as f32) < min_wasted_resolution)
        {
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution as f32;
            best_fit = Some((*height, *width));
        }
    }

    best_fit.unwrap_or((original_height, original_width))
}

fn get_unpadded_features(
    height: usize,
    width: usize,
    npatches: usize,
    num_patch_height: usize,
    num_patch_width: usize,
) -> (usize, usize) {
    let current_height = npatches * num_patch_height;
    let current_width = npatches * num_patch_width;

    let aspect_ratio: f64 = width as f64 / height as f64;
    let current_aspect_ratio: f64 = current_width as f64 / current_height as f64;
    let (current_height, current_width) = if aspect_ratio > current_aspect_ratio {
        let new_height = (height * current_width) / width;
        let padding = (current_height - new_height) / 2;
        (current_height - (2 * padding), current_width)
    } else {
        let new_width = (width * current_height) / height;
        let padding = (current_width - new_width) / 2;
        (current_height, current_width - (2 * padding))
    };

    let unpadded_features = current_height * current_width;
    let newline_features = current_height;
    (unpadded_features, newline_features)
}

impl LlavaNext {
    pub fn get_number_of_features(&self, height: usize, width: usize) -> usize {
        let image_size = self.vision_config.image_size;
        let patch_size = self.vision_config.patch_size;
        assert!(image_size % patch_size == 0);
        let npatches = image_size / patch_size;
        // Dimensions are intentionally swapped to be bug-compatible with
        // upstream: https://github.com/LLaVA-VL/LLaVA-NeXT/issues/59
        let (num_patch_width, num_patch_height) =
            get_anyres_image_grid_shape(height, width, &self.image_grid_pinpoints, image_size);

        let (unpadded_features, newline_features) =
            get_unpadded_features(height, width, npatches, num_patch_height, num_patch_width);
        // The base patch covers the entire image
        let base_features = npatches.pow(2);
        unpadded_features + newline_features + base_features
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Llama4VisionConfig {
    image_size: usize,
    patch_size: usize,
    pixel_shuffle_ratio: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Llama4 {
    text_config: TextConfig,
    vision_config: Llama4VisionConfig,
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn get_factors(dividend: usize) -> HashSet<usize> {
    let mut factors_set = HashSet::new();

    for i in 1..=((dividend as f64).sqrt() as usize) {
        if dividend % i == 0 {
            factors_set.insert(i);
            factors_set.insert(dividend / i);
        }
    }

    factors_set
}

fn find_supported_resolutions(max_num_chunks: usize, height: usize) -> Vec<(usize, usize)> {
    let patch_size = height;

    let mut asp_dict: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();

    for chunk_size in (1..=max_num_chunks).rev() {
        let mut _factors: Vec<_> = get_factors(chunk_size).into_iter().collect();
        _factors.sort();
        let _asp_ratios: Vec<(usize, usize)> =
            _factors.iter().map(|&f| (f, chunk_size / f)).collect();

        for (h, w) in _asp_ratios {
            let divisor = gcd(h, w);
            let key = (h / divisor, w / divisor); // reduced aspect ratio as key

            asp_dict.entry(key).or_default().push((h, w));
        }
    }

    let mut possible_resolutions = vec![];

    for (_key, value) in asp_dict {
        for (h, w) in value {
            possible_resolutions.push((h * patch_size, w * patch_size));
        }
    }

    possible_resolutions
}

fn get_best_fit(
    original_height: usize,
    original_width: usize,
    possible_resolutions: &[(usize, usize)],
    resize_to_max_canvas: bool,
) -> (usize, usize) {
    let orig_h = original_height as f32;
    let orig_w = original_width as f32;

    let mut scales = Vec::with_capacity(possible_resolutions.len());

    for &(h, w) in possible_resolutions.iter() {
        let scale_h = h as f32 / orig_h;
        let scale_w = w as f32 / orig_w;
        let scale = scale_h.min(scale_w);
        scales.push(scale);
    }

    let upscaling_options: Vec<f32> = scales.iter().copied().filter(|&s| s >= 1.0).collect();
    let selected_scale = if !upscaling_options.is_empty() {
        if resize_to_max_canvas {
            upscaling_options.into_iter().fold(f32::MIN, f32::max)
        } else {
            upscaling_options.into_iter().fold(f32::MAX, f32::min)
        }
    } else {
        let downscaling_options: Vec<f32> = scales.iter().copied().filter(|&s| s < 1.0).collect();
        downscaling_options.into_iter().fold(f32::MIN, f32::max)
    };

    let chosen_canvas: Vec<(usize, usize)> = possible_resolutions
        .iter()
        .zip(scales.iter())
        .filter(|&(_, &s)| (s - selected_scale).abs() < f32::EPSILON)
        .map(|(&(h, w), _)| (h, w))
        .collect();

    if chosen_canvas.len() > 1 {
        chosen_canvas
            .into_iter()
            .min_by_key(|(h, w)| h * w)
            .unwrap()
    } else {
        chosen_canvas[0]
    }
}

impl Llama4 {
    pub fn image_size(&self) -> usize {
        self.vision_config.image_size
    }

    pub fn patch_size(&self) -> usize {
        self.vision_config.patch_size
    }

    pub fn pixel_shuffle_ratio(&self) -> f64 {
        self.vision_config.pixel_shuffle_ratio
    }
    pub fn get_aspect_ratios(
        &self,
        height: usize,
        width: usize,
        max_chunks: usize,
    ) -> (usize, usize) {
        let patch_size = self.vision_config.image_size;
        let supported = find_supported_resolutions(max_chunks, patch_size);
        let (target_h, target_w) = get_best_fit(height, width, &supported, false);
        (target_h / patch_size, target_w / patch_size)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ClipVisionModel {
    image_size: usize,
    patch_size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Idefics3 {}

impl Idefics3 {
    pub fn get_max_longest_edge(&self) -> usize {
        364
    }

    pub fn get_number_of_features(&self) -> usize {
        169
    }

    pub fn get_max_longest_edge_for_image_resize(&self) -> usize {
        1456
    }

    pub fn get_max_image_size(&self) -> usize {
        4096
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Idefics2 {}

impl Idefics2 {
    pub fn get_number_of_features(&self, _height: usize, _width: usize) -> usize {
        64
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PaliTextConfig {
    pub(crate) num_image_tokens: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Paligemma {
    pub(crate) text_config: PaliTextConfig,
}

impl Paligemma {
    pub fn get_number_of_features(&self, _height: usize, _width: usize) -> usize {
        self.text_config.num_image_tokens
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Qwen2VlVisionConfig {
    pub(crate) depth: usize,
    pub(crate) embed_dim: usize,
    pub(crate) mlp_ratio: usize,
    pub(crate) num_heads: usize,
    pub(crate) in_chans: usize,
    pub(crate) hidden_size: usize,
    pub(crate) patch_size: usize,
    pub(crate) spatial_merge_size: usize,
    pub(crate) spatial_patch_size: usize,
    pub(crate) temporal_patch_size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Qwen2Vl {
    pub(crate) vision_config: Qwen2VlVisionConfig,
}

impl Qwen2Vl {
    pub fn get_number_of_features(&self, height: usize, width: usize) -> usize {
        let num_pixels = height * width;
        num_pixels / self.vision_config.patch_size.pow(2)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Qwen2_5VlVisionConfig {
    // pub(crate) depth: usize,
    // pub(crate) hidden_act: String,
    // pub(crate) hidden_size: usize,
    // pub(crate) intermediate_size: usize,
    // pub(crate) num_heads: usize,
    // pub(crate) in_chans: usize,
    // pub(crate) out_hidden_size: usize,
    // pub(crate) patch_size: usize,
    // pub(crate) spatial_merge_size: usize,
    pub(crate) spatial_patch_size: usize,
    // pub(crate) window_size: usize,
    // pub(crate) fullatt_block_indexes: Vec<usize>,
    // pub(crate) tokens_per_second: usize,
    // pub(crate) temporal_patch_size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Qwen2_5Vl {
    pub(crate) vision_config: Qwen2_5VlVisionConfig,
}

impl Qwen2_5Vl {
    pub fn get_number_of_features(&self, height: usize, width: usize) -> usize {
        let num_pixels = height * width;
        num_pixels / self.vision_config.spatial_patch_size.pow(2)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Gemma3VisionConfig {
    pub(crate) image_size: usize,
    pub(crate) patch_size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Gemma3 {
    vision_config: Gemma3VisionConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "model_type")]
#[serde(rename_all = "snake_case")]
pub enum Config {
    Qwen2_5Vl(Qwen2_5Vl),
    Qwen2Vl(Qwen2Vl),
    LlavaNext(LlavaNext),
    ClipVisionModel(ClipVisionModel),
    Mistral,
    Mamba,
    Idefics,
    Mllama,
    Idefics2(Idefics2),
    Idefics3(Idefics3),
    Ssm,
    GptBigcode,
    Granite,
    Santacoder,
    Bloom,
    Mpt,
    Gpt2,
    Gptj,
    GptNeox,
    Phi,
    #[serde(rename = "phi-msft")]
    PhiMsft,
    Phi3,
    Phimoe,
    Llama,
    Llama4(Llama4),
    Baichuan,
    Paligemma(Paligemma),
    Gemma,
    Gemma2,
    Gemma3(Gemma3),
    Gemma3Text,
    Cohere,
    Drbx,
    Falcon,
    Mixtral,
    Starcoder2,
    Qwen2,
    Opt,
    T5,
    DeepseekV2,
    DeepseekV3,
    Qwen3,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct TextConfig {}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct VisionConfig {
    pub(crate) image_size: usize,
    pub(crate) patch_size: usize,
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_llava_next_features() {
        let config = LlavaNext {
            text_config: TextConfig {},
            vision_config: VisionConfig {
                image_size: 336,
                patch_size: 14,
            },
            image_grid_pinpoints: vec![
                (336, 672),
                (672, 336),
                (672, 672),
                (1008, 336),
                (336, 1008),
            ],
        };

        let slots = config.get_number_of_features(20, 20);
        assert_eq!(slots, 1176);
        let slots = config.get_number_of_features(640, 640);
        assert_eq!(slots, 2928);
        let slots = config.get_number_of_features(480, 640);
        assert_eq!(slots, 2340);
        let slots = config.get_number_of_features(899, 1024);
        assert_eq!(slots, 2634);
        let slots = config.get_number_of_features(1024, 899);
        assert_eq!(slots, 2640);
        let slots = config.get_number_of_features(1067, 1600);
        assert_eq!(slots, 2144);
    }

    // =========================================================================
    // Tests for utility functions
    // =========================================================================

    #[test]
    fn test_gcd_basic_cases() {
        // Basic GCD cases
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(8, 12), 4);
        assert_eq!(gcd(17, 13), 1); // Coprime numbers
        assert_eq!(gcd(100, 25), 25);
        assert_eq!(gcd(48, 18), 6);
    }

    #[test]
    fn test_gcd_edge_cases() {
        // Edge cases
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(5, 0), 5);
        assert_eq!(gcd(1, 1), 1);
        assert_eq!(gcd(1, 100), 1);
        assert_eq!(gcd(100, 100), 100);
    }

    #[test]
    fn test_get_factors_basic() {
        let factors = get_factors(12);
        assert!(factors.contains(&1));
        assert!(factors.contains(&2));
        assert!(factors.contains(&3));
        assert!(factors.contains(&4));
        assert!(factors.contains(&6));
        assert!(factors.contains(&12));
        assert_eq!(factors.len(), 6);
    }

    #[test]
    fn test_get_factors_prime() {
        // Prime number should only have 1 and itself
        let factors = get_factors(7);
        assert!(factors.contains(&1));
        assert!(factors.contains(&7));
        assert_eq!(factors.len(), 2);
    }

    #[test]
    fn test_get_factors_perfect_square() {
        // Perfect square
        let factors = get_factors(16);
        assert!(factors.contains(&1));
        assert!(factors.contains(&2));
        assert!(factors.contains(&4));
        assert!(factors.contains(&8));
        assert!(factors.contains(&16));
        assert_eq!(factors.len(), 5);
    }

    #[test]
    fn test_get_factors_one() {
        let factors = get_factors(1);
        assert!(factors.contains(&1));
        assert_eq!(factors.len(), 1);
    }

    #[test]
    fn test_select_best_resolution_exact_match() {
        let possible = vec![(640, 480), (800, 600), (1024, 768)];
        // When original matches a possible resolution exactly
        let result = select_best_resolution(480, 640, &possible);
        assert_eq!(result, (640, 480));
    }

    #[test]
    fn test_select_best_resolution_scaling() {
        let possible = vec![(336, 672), (672, 336), (672, 672)];
        // Should find best fit based on effective resolution
        let result = select_best_resolution(400, 800, &possible);
        assert_eq!(result, (336, 672));
    }

    #[test]
    fn test_select_best_resolution_empty_returns_original() {
        // When no possible resolutions, should return original
        let result = select_best_resolution(100, 200, &[]);
        assert_eq!(result, (100, 200));
    }

    #[test]
    fn test_find_supported_resolutions_basic() {
        let resolutions = find_supported_resolutions(4, 100);
        // Should produce resolutions based on chunk factors
        assert!(!resolutions.is_empty());
        // All resolutions should be multiples of patch_size (100)
        for (h, w) in &resolutions {
            assert_eq!(h % 100, 0);
            assert_eq!(w % 100, 0);
        }
    }

    #[test]
    fn test_find_supported_resolutions_includes_expected() {
        let resolutions = find_supported_resolutions(4, 100);
        // Should include 1x1, 2x1, 1x2, 2x2, 4x1, 1x4 patterns scaled by patch_size
        assert!(resolutions.contains(&(100, 100))); // 1x1
        assert!(resolutions.contains(&(200, 200))); // 2x2
    }

    #[test]
    fn test_get_best_fit_upscaling() {
        let possible = vec![(200, 200), (400, 400), (800, 800)];
        // Small image should upscale to smallest fitting resolution
        let result = get_best_fit(100, 100, &possible, false);
        assert_eq!(result, (200, 200));
    }

    #[test]
    fn test_get_best_fit_upscaling_max_canvas() {
        let possible = vec![(200, 200), (400, 400), (800, 800)];
        // With resize_to_max_canvas=true, should pick largest
        let result = get_best_fit(100, 100, &possible, true);
        assert_eq!(result, (800, 800));
    }

    #[test]
    fn test_get_best_fit_downscaling() {
        let possible = vec![(200, 200), (400, 400)];
        // Large image should downscale to largest fitting resolution
        let result = get_best_fit(1000, 1000, &possible, false);
        assert_eq!(result, (400, 400));
    }

    // =========================================================================
    // Tests for model-specific configurations
    // =========================================================================

    #[test]
    fn test_idefics2_constant_features() {
        let config = Idefics2 {};
        // Idefics2 always returns 64 features regardless of dimensions
        assert_eq!(config.get_number_of_features(100, 100), 64);
        assert_eq!(config.get_number_of_features(1000, 500), 64);
        assert_eq!(config.get_number_of_features(1, 1), 64);
    }

    #[test]
    fn test_idefics3_constants() {
        let config = Idefics3 {};
        assert_eq!(config.get_max_longest_edge(), 364);
        assert_eq!(config.get_number_of_features(), 169);
        assert_eq!(config.get_max_longest_edge_for_image_resize(), 1456);
        assert_eq!(config.get_max_image_size(), 4096);
    }

    #[test]
    fn test_paligemma_features_from_config() {
        let config = Paligemma {
            text_config: PaliTextConfig {
                num_image_tokens: 256,
            },
        };
        // Paligemma returns num_image_tokens regardless of dimensions
        assert_eq!(config.get_number_of_features(100, 100), 256);
        assert_eq!(config.get_number_of_features(500, 300), 256);
    }

    #[test]
    fn test_qwen2vl_features_calculation() {
        let config = Qwen2Vl {
            vision_config: Qwen2VlVisionConfig {
                depth: 24,
                embed_dim: 1024,
                mlp_ratio: 4,
                num_heads: 16,
                in_chans: 3,
                hidden_size: 1024,
                patch_size: 14,
                spatial_merge_size: 2,
                spatial_patch_size: 14,
                temporal_patch_size: 2,
            },
        };
        // Features = (height * width) / patch_size^2
        // 196 * 196 / 14^2 = 38416 / 196 = 196
        assert_eq!(config.get_number_of_features(196, 196), 196);
        // 280 * 280 / 14^2 = 78400 / 196 = 400
        assert_eq!(config.get_number_of_features(280, 280), 400);
    }

    #[test]
    fn test_qwen2_5vl_features_calculation() {
        let config = Qwen2_5Vl {
            vision_config: Qwen2_5VlVisionConfig {
                spatial_patch_size: 14,
            },
        };
        // Features = (height * width) / spatial_patch_size^2
        assert_eq!(config.get_number_of_features(196, 196), 196);
        assert_eq!(config.get_number_of_features(280, 280), 400);
    }

    #[test]
    fn test_llama4_accessors() {
        let config = Llama4 {
            text_config: TextConfig {},
            vision_config: Llama4VisionConfig {
                image_size: 560,
                patch_size: 14,
                pixel_shuffle_ratio: 0.5,
            },
        };
        assert_eq!(config.image_size(), 560);
        assert_eq!(config.patch_size(), 14);
        assert!((config.pixel_shuffle_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_llama4_aspect_ratios() {
        let config = Llama4 {
            text_config: TextConfig {},
            vision_config: Llama4VisionConfig {
                image_size: 560,
                patch_size: 14,
                pixel_shuffle_ratio: 0.5,
            },
        };
        // Test aspect ratio calculation
        let (h, w) = config.get_aspect_ratios(560, 560, 4);
        assert!(h > 0 && w > 0);
    }

    // =========================================================================
    // Tests for Config enum serialization
    // =========================================================================

    #[test]
    fn test_config_deserialize_simple_variants() {
        // Test simple variants without data
        let json = r#"{"model_type": "mistral"}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert!(matches!(config, Config::Mistral));

        let json = r#"{"model_type": "llama"}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert!(matches!(config, Config::Llama));

        let json = r#"{"model_type": "gemma2"}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert!(matches!(config, Config::Gemma2));
    }

    #[test]
    fn test_config_deserialize_idefics2() {
        let json = r#"{"model_type": "idefics2"}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert!(matches!(config, Config::Idefics2(_)));
    }

    #[test]
    fn test_config_deserialize_idefics3() {
        let json = r#"{"model_type": "idefics3"}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert!(matches!(config, Config::Idefics3(_)));
    }

    #[test]
    fn test_config_deserialize_qwen2vl() {
        let json = r#"{
            "model_type": "qwen2_vl",
            "vision_config": {
                "depth": 24,
                "embed_dim": 1024,
                "mlp_ratio": 4,
                "num_heads": 16,
                "in_chans": 3,
                "hidden_size": 1024,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "spatial_patch_size": 14,
                "temporal_patch_size": 2
            }
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert!(matches!(config, Config::Qwen2Vl(_)));
    }

    #[test]
    fn test_config_deserialize_deepseek_variants() {
        let json = r#"{"model_type": "deepseek_v2"}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert!(matches!(config, Config::DeepseekV2));

        let json = r#"{"model_type": "deepseek_v3"}"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert!(matches!(config, Config::DeepseekV3));
    }
}

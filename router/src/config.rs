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
}

use serde::{Deserialize, Serialize};

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
pub struct ClipVisionModel {
    image_size: usize,
    patch_size: usize,
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
#[serde(tag = "model_type")]
#[serde(rename_all = "snake_case")]
pub enum Config {
    LlavaNext(LlavaNext),
    ClipVisionModel(ClipVisionModel),
    Mistral,
    Idefics,
    Idefics2(Idefics2),
    Ssm,
    GptBigcode,
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
    Llama,
    Baichuan,
    Paligemma(Paligemma),
    Gemma,
    Gemma2,
    Cohere,
    Drbx,
    Falcon,
    Mixtral,
    Starcoder2,
    Qwen2,
    Opt,
    T5,
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

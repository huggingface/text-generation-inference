use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "model_type")]
#[serde(rename_all = "snake_case")]
pub struct LlavaNext {
    text_config: TextConfig,
    vision_config: VisionConfig,
    image_grid_pinpoints: Vec<(usize, usize)>,
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

impl LlavaNext {
    pub fn get_number_of_features(&self, height: usize, width: usize) -> usize {
        let image_size = self.vision_config.image_size;
        let patch_size = self.vision_config.patch_size;
        assert!(image_size % patch_size == 0);
        let npatches = image_size / patch_size;
        let (num_patch_height, num_patch_width) =
            get_anyres_image_grid_shape(height, width, &self.image_grid_pinpoints, image_size);
        // Ceil
        let height_of_patch = (height * npatches + width - 1) / width;
        let unpadded_features = npatches * height_of_patch * num_patch_height * num_patch_width;
        // They are only added after width
        let newline_features = height_of_patch * num_patch_width;
        // The base patch covers the entire image
        let base_features = npatches.pow(2);
        unpadded_features + newline_features + base_features
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "model_type")]
#[serde(rename_all = "snake_case")]
pub struct ClipVisionModel {
    image_size: usize,
    patch_size: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "model_type")]
#[serde(rename_all = "snake_case")]
pub enum Config {
    LlavaNext(LlavaNext),
    ClipVisionModel(ClipVisionModel),
    Mistral,
    Idefics,
    Ssm,
    GptBigcode,
    Santacoder,
    Bloom,
    Mpt,
    GptNeox,
    Phi,
    #[serde(rename = "phi-msft")]
    PhiMsft,
    Llama,
    Baichuan,
    Gemma,
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
    image_size: usize,
    patch_size: usize,
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

        let slots = config.get_number_of_features(640, 640);
        assert_eq!(slots, 2928);
        let slots = config.get_number_of_features(480, 640);
        assert_eq!(slots, 2340);
        let slots = config.get_number_of_features(899, 1024);
        assert_eq!(slots, 2732);
        let slots = config.get_number_of_features(1024, 899);
        assert_eq!(slots, 3320);
    }
}

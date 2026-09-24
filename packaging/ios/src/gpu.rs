//! The limits the window's graphics device is asked for.
//!
//! `egui-wgpu` asks for `wgpu::Limits::default()`, the WebGPU defaults, and
//! one of them is more than any iPhone or iPad has: 16 inter-stage shader
//! variables. wgpu's Metal backend reports 124 varying components (31
//! variables) only on a Mac and 60 (15 variables) on every iOS GPU and in
//! the iOS simulator, and a device request above what the adapter offers
//! fails - the program stopped before its first frame with "Limit
//! 'max_inter_stage_shader_variables' value 16 is better than allowed 15",
//! which is what the first simulator run in CI showed. egui's own shaders
//! pass three values between stages.
//!
//! So the request is the same one `egui-wgpu` makes (the WebGPU defaults,
//! textures up to 8192 for a depth buffer the size of a large screen),
//! lowered to the adapter's own limits wherever the adapter offers less.
//! The inference engines are not affected: `cubecl` asks for the adapter's
//! limits as they are.

use wgpu_types::Limits;

/// What to request from an adapter offering `adapter`.
pub fn device_limits(adapter: &Limits) -> Limits {
    Limits {
        max_texture_dimension_2d: 8192,
        ..Limits::default()
    }
    .or_worse_values_from(adapter)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The limits wgpu's Metal backend reports on an iPad or iPhone, as far
    /// as they matter here.
    fn ios_gpu() -> Limits {
        Limits {
            max_inter_stage_shader_variables: 15,
            max_texture_dimension_2d: 16384,
            ..Limits::default()
        }
    }

    #[test]
    fn an_ios_gpu_is_asked_for_no_more_than_it_has() {
        let got = device_limits(&ios_gpu());
        assert_eq!(got.max_inter_stage_shader_variables, 15);
        assert_eq!(got.max_texture_dimension_2d, 8192);
        // Everything else stays the WebGPU default the desktop asks for.
        assert_eq!(got.max_bind_groups, Limits::default().max_bind_groups);
        assert!(got.check_limits(&ios_gpu()));
    }

    #[test]
    fn a_smaller_texture_limit_is_taken_as_it_is() {
        let small = Limits {
            max_texture_dimension_2d: 4096,
            ..ios_gpu()
        };
        assert_eq!(device_limits(&small).max_texture_dimension_2d, 4096);
    }
}

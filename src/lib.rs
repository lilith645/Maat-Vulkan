pub extern crate winit;
pub extern crate cgmath;
pub extern crate vk;
pub extern crate image;

pub use self::window::VkWindow;
pub use self::swapchain::Swapchain;
pub use self::shader::Shader;
pub use self::instance::Instance;
pub use self::device::Device;
pub use self::descriptorset::DescriptorSet;
pub use self::descriptorset::UpdateDescriptorSets;
pub use self::descriptorset::DescriptorSetBuilder;
pub use self::pipeline::Pipeline;
pub use self::pipeline::PipelineInfo;
pub use self::pipelinebuilder::PipelineBuilder;
pub use self::renderpass::RenderPass;
pub use self::renderpassbuilder::RenderPassBuilder;
pub use self::renderpassbuilder::AttachmentInfo;
pub use self::renderpassbuilder::SubpassInfo;
pub use self::ownage::check_errors;
pub use self::sampler::Sampler;
pub use self::sampler::SamplerBuilder;
pub use self::imageattachment::ImageAttachment;
pub use self::clearvalues::ClearValues;
pub use self::compute::Compute;
pub use self::settings::Settings;
pub use self::logs::Logs;

pub mod pool;
pub mod buffer;
pub mod sync;
pub mod vkenums;
mod settings;
mod loader;
mod ownage;
mod window;
mod swapchain;
mod shader;
mod instance;
mod device;
mod descriptorset;
mod pipeline;
mod pipelinebuilder;
mod renderpass;
mod renderpassbuilder;
mod sampler;
mod imageattachment;
mod clearvalues;
mod compute;
mod logs;

const ENGINE_VERSION: u32 = (0 as u32) << 22 | (2 as u32) << 12 | (0 as u32);

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

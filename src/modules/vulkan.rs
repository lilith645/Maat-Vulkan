use ash::extensions::{
    khr::{Surface, Swapchain},
};

use ash::version::{DeviceV1_0};
use ash::{vk};
use std::default::Default;
use winit::window::Window;

use crate::modules::{VkDevice, VkInstance, VkCommandPool, VkSwapchain, VkFrameBuffer, Scissors, 
                     ClearValues, Viewport, Fence, Semaphore, ImageBuilder, Image, Renderpass, 
                     PassDescription, VkWindow, Buffer, GraphicsPipeline, Shader, DescriptorSet,
                     ComputeShader, DescriptorWriter};

// Simple offset_of macro akin to C++ offsetof
#[macro_export]
macro_rules! offset_of {
  ($base:path, $field:ident) => {{
    #[allow(unused_unsafe)]
    unsafe {
      let b: $base = mem::zeroed();
      (&b.$field as *const _ as isize) - (&b as *const _ as isize)
    }
  }};
}

pub fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _memory_type)| index as _)
}

pub struct Vulkan {
  instance: VkInstance,
  device: VkDevice,
  
  renderpass: Renderpass,
  framebuffer: VkFrameBuffer,
  
  swapchain: VkSwapchain,
  
  pool: VkCommandPool,
  
  pub draw_command_buffer: vk::CommandBuffer,
  pub setup_command_buffer: vk::CommandBuffer,
  
  depth_image: Image,

  present_complete_semaphore: Semaphore,
  rendering_complete_semaphore: Semaphore,

  draw_commands_reuse_fence: Fence,
  setup_commands_reuse_fence: Fence,
  
  scissors: Scissors,
  clear_values: ClearValues,
  viewports: Viewport,
}

impl Vulkan {
  pub fn new(window: &mut VkWindow, screen_resolution: vk::Extent2D) -> Vulkan {
    
    let instance = VkInstance::new(window);
    let device = VkDevice::new(&instance, window);
    
    let mut swapchain = VkSwapchain::new(&instance, &device, screen_resolution);
    
    let pool = VkCommandPool::new(&device);
    let command_buffers = pool.allocate_primary_command_buffers(&device, 2);
    
    let setup_command_buffer = command_buffers[0];
    let draw_command_buffer = command_buffers[1];
    
    let extent = swapchain.extent();
    let depth_image = ImageBuilder::new_depth(extent.width, extent.height,
                                              1, 1, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                                    .build_device_local(&device);
    let passes = vec![
      PassDescription::new(device.surface_format().format)
                       .samples_1()
                       .attachment_load_op_clear()
                       .attachment_store_op_store()
                       .attachment_layout_colour()
                       .final_layout_present_src(),
      PassDescription::new(vk::Format::D16_UNORM)
                       .samples_1()
                       .attachment_load_op_clear()
                       .attachment_layout_depth_stencil()
                       .initial_layout_undefined()
                       .final_layout_depth_stencil()
    ];
    
    let renderpass = Renderpass::new(&device, passes);
    
    let framebuffer = VkFrameBuffer::new(&device, &mut swapchain, &depth_image, &renderpass);
    
    let draw_commands_reuse_fence = Fence::new_signaled(&device);
    let setup_commands_reuse_fence = Fence::new_signaled(&device);
    
    let present_complete_semaphore = Semaphore::new(&device);
    let rendering_complete_semaphore = Semaphore::new(&device);
    
    let clear_values = ClearValues::new().add_colour(0.0, 0.0, 0.0, 0.0).add_depth(1.0, 0);
    let scissors = Scissors::new().add_scissor(0, 0, extent.width, extent.height);
    
    let viewports = Viewport::new(0.0, 0.0, 
                                  extent.width as f32,
                                  extent.height as f32,
                                  0.0, 1.0);

    Vulkan {
        instance,
        device,
        renderpass,
        swapchain,
        pool,
        
        draw_command_buffer,
        setup_command_buffer,
        depth_image,
        present_complete_semaphore,
        rendering_complete_semaphore,
        draw_commands_reuse_fence,
        setup_commands_reuse_fence,
        viewports,
        framebuffer,
        scissors,
        clear_values,
    }
  }
  
  pub fn swapchain(&mut self) -> &mut VkSwapchain {
    &mut self.swapchain
  }
  
  pub fn renderpass(&self) -> &Renderpass {
    &self.renderpass
  }
  
  pub fn scissors(&self) -> &Scissors {
    &self.scissors
  }
  
  pub fn viewports(&self) -> &Viewport {
    &self.viewports
  }
  
  pub fn recreate_swapchain(&mut self) {
    unsafe {
        let device = self.device.internal();
        
        device.device_wait_idle().unwrap();
        
        self.framebuffer.destroy(device);

        device.destroy_image_view(self.depth_image.view(), None);
        device.destroy_image(self.depth_image.internal(), None);
        device.free_memory(self.depth_image.memory(), None);
    }
    self.swapchain.destroy(&self.device);
    
    self.swapchain.recreate(&self.instance, &self.device);
    let extent = self.swapchain.extent();
    
    self.depth_image = ImageBuilder::new_depth(extent.width, extent.height,
                                              1, 1, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                                     .build_device_local(&self.device);
    
    self.framebuffer = VkFrameBuffer::new(&self.device, &mut self.swapchain, &self.depth_image, &self.renderpass);

    self.scissors = Scissors::new().add_scissor(0, 0, extent.width, extent.height);

    self.viewports = Viewport::new(0.0, 0.0, 
                                   extent.width as f32,
                                   extent.height as f32,
                                   0.0, 1.0);
  }

  pub fn render_triangle<T: Copy, L: Copy>(
      &mut self,
      vertex_buffer: &Buffer<T>,
      index_buffer: &Buffer<L>,
      graphics_pipeline: &GraphicsPipeline,
  ) {
    let present_index_result = unsafe {
      self.swapchain.swapchain_loader()
          .acquire_next_image(
              *self.swapchain.internal(),
              std::u64::MAX,
              self.present_complete_semaphore.internal(),
              vk::Fence::null(),
          )
    };
    
    let (present_index, _) = match present_index_result {
      Ok(index) => index,
      Err(e) => {
          self.recreate_swapchain();
          return;
      }
    };
    
    let clear_values = self.clear_values.build();
    let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
        .render_pass(self.renderpass.internal())
        .framebuffer(self.framebuffer.framebuffers()[present_index as usize])
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain.extent(),
        })
        .clear_values(&clear_values);

    Vulkan::record_submit_commandbuffer(
      &self.device,
      self.draw_command_buffer,
      &self.draw_commands_reuse_fence,
      self.device.present_queue(),
      &[vk::PipelineStageFlags::BOTTOM_OF_PIPE],
      &self.present_complete_semaphore,
      &self.rendering_complete_semaphore,
      |device, draw_command_buffer| { unsafe {
        device.cmd_begin_render_pass(
          draw_command_buffer,
          &render_pass_begin_info,
          vk::SubpassContents::INLINE,
        );
        
        device.cmd_bind_pipeline(
          draw_command_buffer,
          vk::PipelineBindPoint::GRAPHICS,
          *graphics_pipeline.internal(),
        );
        
        device.cmd_set_viewport(draw_command_buffer, 0, &[self.viewports.build()]);
        device.cmd_set_scissor(draw_command_buffer, 0, &self.scissors.build());
        device.cmd_bind_vertex_buffers(
          draw_command_buffer,
          0,
          &[*vertex_buffer.internal()],
          &[0],
        );
        device.cmd_bind_index_buffer(
          draw_command_buffer,
          *index_buffer.internal(),
          0,
          vk::IndexType::UINT32,
        );
        device.cmd_draw_indexed(
          draw_command_buffer,
          index_buffer.data().len() as u32,
          1,
          0,
          0,
          1,
        );
        // Or draw without the index buffer
        // device.cmd_draw(draw_command_buffer, 3, 1, 0, 0);
        device.cmd_end_render_pass(draw_command_buffer);
      }},
    );

    let wait_semaphores = [self.rendering_complete_semaphore.internal()];
    let swapchains = [*self.swapchain.internal()];
    let image_indices = [present_index];
    let present_info = vk::PresentInfoKHR::builder()
        .wait_semaphores(&wait_semaphores)
        .swapchains(&swapchains)
        .image_indices(&image_indices);

    unsafe {
        match self.swapchain.swapchain_loader()
            .queue_present(self.device.present_queue(), &present_info) {
        Ok(_) => {
          
        },
        Err(vk_e) => {
          match vk_e {
            vk::Result::ERROR_OUT_OF_DATE_KHR => { //VK_ERROR_OUT_OF_DATE_KHR
              self.recreate_swapchain();
              return;
            },
            e => {
              panic!("Error: {}", e);
            }
          }
        }
      }
    };
  }
  
  pub fn render_texture<T: Copy, L: Copy>(
    &mut self,
    descriptor_sets: &DescriptorSet,//&Vec<vk::DescriptorSet>,
    shader: &Shader<T>,
    vertex_buffer: &Buffer<T>,
    index_buffer: &Buffer<L>,
  ) {
    let present_index_result = unsafe {
      self.swapchain.swapchain_loader()
          .acquire_next_image(
              *self.swapchain.internal(),
              std::u64::MAX,
              self.present_complete_semaphore.internal(),
              vk::Fence::null(),
          )
    };
    let (present_index, _) = match present_index_result {
      Ok(index) => index,
      Err(_) => {
        self.recreate_swapchain();
        return;
      }
    };
    
    let clear_values = self.clear_values.build();
    let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
        .render_pass(self.renderpass.internal())
        .framebuffer(self.framebuffer.framebuffers()[present_index as usize])
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain.extent(),
        })
        .clear_values(&clear_values);

    Vulkan::record_submit_commandbuffer(
      &self.device,
      self.draw_command_buffer,
      &self.draw_commands_reuse_fence,
      self.device.present_queue(),
      &[vk::PipelineStageFlags::BOTTOM_OF_PIPE],
      &self.present_complete_semaphore,
      &self.rendering_complete_semaphore,
      |device, draw_command_buffer| { unsafe {
        device.cmd_begin_render_pass(
          draw_command_buffer,
          &render_pass_begin_info,
          vk::SubpassContents::INLINE,
        );
        
        device.cmd_bind_descriptor_sets(
          draw_command_buffer,
          vk::PipelineBindPoint::GRAPHICS,
          shader.pipeline_layout(),
          0,
          &descriptor_sets.internal()[..],
          &[],
        );
        
        device.cmd_bind_pipeline(
          draw_command_buffer,
          vk::PipelineBindPoint::GRAPHICS,
          *shader.graphics_pipeline().internal(),
        );
        
        device.cmd_set_viewport(draw_command_buffer, 0, &[self.viewports.build()]);
        device.cmd_set_scissor(draw_command_buffer, 0, &self.scissors.build());
        
        device.cmd_bind_vertex_buffers(
          draw_command_buffer,
          0,
          &[*vertex_buffer.internal()],
          &[0],
        );
        
        device.cmd_bind_index_buffer(
          draw_command_buffer,
          *index_buffer.internal(),
          0,
          vk::IndexType::UINT32,
        );
        
        device.cmd_draw_indexed(
          draw_command_buffer,
          index_buffer.data().len() as u32,
          1,
          0,
          0,
          1,
        );
        
        // Or draw without the index buffer
        // device.cmd_draw(draw_command_buffer, 3, 1, 0, 0);
        device.cmd_end_render_pass(draw_command_buffer);
      }},
    );

    let present_info = vk::PresentInfoKHR {
      wait_semaphore_count: 1,
      p_wait_semaphores: &self.rendering_complete_semaphore.internal(),
      swapchain_count: 1,
      p_swapchains: self.swapchain.internal(),
      p_image_indices: &present_index,
      ..Default::default()
    };
    
    unsafe {
        match self.swapchain.swapchain_loader()
            .queue_present(self.device.present_queue(), &present_info) {
        Ok(_) => {
          
        },
        Err(vk_e) => {
          match vk_e {
            vk::Result::ERROR_OUT_OF_DATE_KHR => { //VK_ERROR_OUT_OF_DATE_KHR
              self.recreate_swapchain();
              return;
            },
            e => {
              panic!("Error: {}", e);
            }
          }
        }
      }
    };
  }
  
  pub fn copy_buffer_to_device_local_image(&mut self, src_buffer: &Buffer<u8>, dst_image: &Image) {
    Vulkan::record_submit_commandbuffer(
      &self.device,
      self.setup_command_buffer,
      &self.setup_commands_reuse_fence,
      self.device.present_queue(),
      &[],
      &Semaphore::new(&self.device),//[],
      &Semaphore::new(&self.device),//[],
      |device, texture_command_buffer| {
        let texture_barrier = vk::ImageMemoryBarrier {
          dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
          new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
          image: dst_image.internal(),
          subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            level_count: 1,
            layer_count: 1,
            ..Default::default()
          },
          ..Default::default()
        };
        
        unsafe {
          device.cmd_pipeline_barrier(
            texture_command_buffer,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[texture_barrier],
          );
        }
        
        let buffer_copy_regions = vk::BufferImageCopy::builder()
            .image_subresource(
              vk::ImageSubresourceLayers::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1)
                .build(),
            )
            .image_extent(vk::Extent3D {
              width: dst_image.width(),//image_dimensions.0,
              height: dst_image.height(),//image_dimensions.1,
              depth: 1,
            });
        
        unsafe {
          device.cmd_copy_buffer_to_image(
            texture_command_buffer,
            *src_buffer.internal(),
            dst_image.internal(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[buffer_copy_regions.build()],
          );
        }
        
        let texture_barrier_end = vk::ImageMemoryBarrier {
          src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
          dst_access_mask: vk::AccessFlags::SHADER_READ,
          old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
          new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
          image: dst_image.internal(),
          subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            level_count: 1,
            layer_count: 1,
            ..Default::default()
          },
          ..Default::default()
        };
        
        unsafe {
          device.cmd_pipeline_barrier(
            texture_command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[texture_barrier_end],
          );
        }
      },
    );
  }
  
  pub fn copy_buffer_to_device_local_buffer<T: Copy>(&mut self, src_buffer: &Buffer<T>, dst_buffer: &Buffer<T>) {
    Vulkan::record_submit_commandbuffer(
      &self.device,
      self.setup_command_buffer,
      &self.setup_commands_reuse_fence,
      self.device.present_queue(),
      &[],
      &Semaphore::new(&self.device),//[],
      &Semaphore::new(&self.device),//[],
      |device, buffer_command_buffer| {
        /*let buffer_barrier = vk::BufferMemoryBarrier {
          dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
          buffer: dst_buffer.internal(),
          size: std::mem::size_of::<T>() as u64 * (src_buffer.data().len() as u64)
          ..Default::default()
        };*/
        let buffer_barrier = vk::BufferMemoryBarrier::builder()
                                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                                .buffer(*dst_buffer.internal())
                                .size(std::mem::size_of::<T>() as u64 * (src_buffer.data().len() as u64));
        
        unsafe {
          device.cmd_pipeline_barrier(
            buffer_command_buffer,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[buffer_barrier.build()],
            &[],
          );
        }
        
        let buffer_copy = vk::BufferCopy::builder()
                             .size(std::mem::size_of::<T>() as u64 * (src_buffer.data().len() as u64));
        /*
        let buffer_copy_regions = vk::BufferImageCopy::builder()
            .image_subresource(
              vk::ImageSubresourceLayers::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1)
                .build(),
            )
            .image_extent(vk::Extent3D {
              width: dst_image.width(),//image_dimensions.0,
              height: dst_image.height(),//image_dimensions.1,
              depth: 1,
            });*/
        
        unsafe {
          device.cmd_copy_buffer(
            buffer_command_buffer,
            *src_buffer.internal(),
            *dst_buffer.internal(),
            &[buffer_copy.build()],
          );
          /*device.cmd_copy_buffer_to_image(
            texture_command_buffer,
            *src_buffer.internal(),
            dst_image.internal(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[buffer_copy_regions.build()],
          );*/
        }
        
        let buffer_barrier_end = vk::BufferMemoryBarrier::builder()
                                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                                    .buffer(*dst_buffer.internal())
                                    .size(std::mem::size_of::<T>() as u64 * (src_buffer.data().len() as u64));
        /*
        let buffer_barrier_end = vk::BufferMemoryBarrier {
          src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
          dst_access_mask: vk::AccessFlags::SHADER_READ,
          buffer: dst_buffer.internal(),
          ..Default::default()
        };*/
        
        unsafe {
          device.cmd_pipeline_barrier(
            buffer_command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[buffer_barrier_end.build()],
            &[],
          );
        }
      },
    );
  }
  
  pub fn run_compute<T: Copy>(&mut self, compute_shader: &ComputeShader, 
                              descriptor_sets: &DescriptorSet, data: &mut Vec<T>) {
    let src_buffer = Buffer::<T>::builder()
                                     .data(data.to_vec())
                                     .usage_transfer_src_dst()
                                     .memory_properties_host_visible_coherent()
                                     .build(&self.device);
    let dst_buffer = Buffer::<T>::builder()
                                     .data(data.to_vec())
                                     .usage_transfer_storage_src_dst()
                                     .memory_properties_host_visible_coherent()
                                     .build(&self.device);
    
    let descriptor_set_writer = DescriptorWriter::builder()
                                                .update_storage_buffer(&dst_buffer, 
                                                                       &descriptor_sets);
    descriptor_set_writer.build(&self.device);
    
    Vulkan::record_submit_commandbuffer(
      &self.device,
      self.setup_command_buffer,
      &self.setup_commands_reuse_fence,
      self.device.present_queue(),
      &[],
      &Semaphore::new(&self.device),//[],
      &Semaphore::new(&self.device),//[],
      |device, compute_command_buffer| {
        
        let buffer_copy = vk::BufferCopy::builder()
                             .size(std::mem::size_of::<T>() as u64 * (src_buffer.data().len() as u64));
        
        unsafe {
          device.cmd_copy_buffer(
            compute_command_buffer,
            *src_buffer.internal(),
            *dst_buffer.internal(),
            &[buffer_copy.build()],
          );
        }
        
        let buffer_barrier = vk::BufferMemoryBarrier::builder()
                                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                                .buffer(*dst_buffer.internal())
                                .size(std::mem::size_of::<T>() as u64 * (src_buffer.data().len() as u64));
        
        unsafe {
          device.cmd_pipeline_barrier(
            compute_command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[buffer_barrier.build()],
            &[],
          );
        }
        
        unsafe {
          device.cmd_bind_pipeline(
            compute_command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            *compute_shader.pipeline().internal(),
          );
          
          device.cmd_bind_descriptor_sets(
            compute_command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            compute_shader.pipeline_layout(),
            0,
            &descriptor_sets.internal()[..],
            &[],
          );
        }
        
        unsafe {
          device.cmd_dispatch(
            compute_command_buffer,
            src_buffer.data().len() as u32,
            1,
            1,
          )
        }
        
        let buffer_barrier_end = vk::BufferMemoryBarrier::builder()
                                    .src_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                                    .buffer(*dst_buffer.internal())
                                    .size(std::mem::size_of::<T>() as u64 * (data.len() as u64));
        
        unsafe {
          device.cmd_pipeline_barrier(
            compute_command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[buffer_barrier_end.build()],
            &[],
          );
        }
        
        let buffer_copy = vk::BufferCopy::builder()
                             .size(std::mem::size_of::<T>() as u64 * (data.len() as u64));
        
        unsafe {
          device.cmd_copy_buffer(
            compute_command_buffer,
            *dst_buffer.internal(),
            *src_buffer.internal(),
            &[buffer_copy.build()],
          );
        }
      },
    );
    
    unsafe { self.device.internal().device_wait_idle().unwrap() };
    *data = src_buffer.retrieve_buffer_data(&self.device);
  }
  
  pub fn device(&self) -> &VkDevice {
    &self.device
  }
  
  /// Helper function for submitting command buffers. Immediately waits for the fence before the command buffer
  /// is executed. That way we can delay the waiting for the fences by 1 frame which is good for performance.
  /// Make sure to create the fence in a signaled state on the first use.
  #[allow(clippy::too_many_arguments)]
  pub fn record_submit_commandbuffer<F: FnOnce(&VkDevice, vk::CommandBuffer)>(
      device: &VkDevice,
      command_buffer: vk::CommandBuffer,
      command_buffer_reuse_fence: &Fence,
      submit_queue: vk::Queue,
      wait_mask: &[vk::PipelineStageFlags],
      wait_semaphores: &Semaphore,
      signal_semaphores: &Semaphore,
      f: F,
  ) {
    unsafe {
        command_buffer_reuse_fence.wait(device);
        command_buffer_reuse_fence.reset(device);
        
        device
            .reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )
            .expect("Reset command buffer failed.");

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .expect("Begin commandbuffer");
        f(device, command_buffer);
        device
            .end_command_buffer(command_buffer)
            .expect("End commandbuffer");

        let command_buffers = vec![command_buffer];
        
        let wait_semaphore = [wait_semaphores.internal()];
        let signal_semaphore = [signal_semaphores.internal()];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphore)
            .wait_dst_stage_mask(wait_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphore);

        device
            .queue_submit(
                submit_queue,
                &[submit_info.build()],
                command_buffer_reuse_fence.internal(),
            )
            .expect("queue submit failed.");
    }
  }
}



use vk;

use crate::vkenums::{ImageLayout, Access, ImageAspect, PipelineStage};

use crate::buffer::{CommandBuffer, CommandBufferBuilder, Buffer};
use crate::pool::{CommandPool, DescriptorPool};
use crate::sync::{Fence};
use crate::{Instance, Device, DescriptorSet, DescriptorSetBuilder, UpdateDescriptorSets, 
                    Pipeline, PipelineBuilder, ImageAttachment, Shader};

use crate::Logs;

use std::sync::Arc;

pub struct Compute {
  queue: vk::Queue,
  _family: u32,
  shader: Shader,
  command_pool: CommandPool,
  command_buffers: Vec<Arc<CommandBuffer>>,
  fences: Vec<Fence>,
  descriptor_sets: Vec<DescriptorSet>,
  pipeline: Pipeline,
}
/*
impl Compute {
  pub fn new(instance: Arc<Instance>, device: Arc<Device>, _dummy_image: &ImageAttachment, descriptor_pool: &DescriptorPool, num_sets: u32) -> Compute {
    let (compute_queue, compute_family) = device.get_compute_queue(Arc::clone(&instance));
    
    let compute_shader = Shader::new(Arc::clone(&device), include_bytes!("../shaders/sprv/ComputeSharpen.spv"));
    
    let mut descriptor_sets = Vec::with_capacity(num_sets as usize);
    
    
    descriptor_sets.push(DescriptorSetBuilder::new()
                         .build(Arc::clone(&device), descriptor_pool, 1));
    
    let pipeline = PipelineBuilder::new()
                     .compute_shader(*compute_shader.get_shader())
                     .descriptor_set_layout(descriptor_sets[0].layouts_clone())
                     .build_compute(Arc::clone(&device));
    
    let command_pool = CommandPool::new(Arc::clone(&device), compute_family);
    let command_buffers = command_pool.create_command_buffers(Arc::clone(&device), num_sets);
    
    let mut fences = Vec::with_capacity(num_sets as usize);
    for _ in 0..num_sets as usize {
      fences.push(Fence::new(Arc::clone(&device)));
    }
    
    Compute {
      queue: compute_queue,
      _family: compute_family,
      shader: compute_shader,
      command_pool,
      command_buffers,
      fences,
      descriptor_sets,
      pipeline,
    }
    Compute {
    
    }
  }
  
  // Running compute shaders
  //
  // vkCmDispatch
  // or vkCmdDispatchIndirect
  
  
}*/

/*
impl Compute {
  pub fn new(instance: Arc<Instance>, device: Arc<Device>, buffer: &Buffer<f32>, descriptor_pool: &DescriptorPool, num_sets: u32, logs: &mut Logs) -> Compute {
    
    let (compute_queue, compute_family) = device.get_compute_queue(Arc::clone(&instance), logs);
    
    let compute_shader = Shader::new(Arc::clone(&device), include_bytes!("../shaders/sprv/ComputeSharpen.spv"));
    
    let mut descriptor_sets = Vec::with_capacity(num_sets as usize);
    
    for _ in 0..num_sets {
      descriptor_sets.push(DescriptorSetBuilder::new()
                           .compute_storage_buffer(0)
                           .build(Arc::clone(&device), descriptor_pool, 1));
    }/*
    UpdateDescriptorSets::new()
      .add_storage_image(0, &dummy_image, ImageLayout::StorageImage)
      .finish_update(Arc::clone(&device), &descriptor_set);*/
    
    let pipeline = PipelineBuilder::new()
                     .compute_shader(*compute_shader.get_shader())
                     .descriptor_set_layout(descriptor_sets[0].layouts_clone())
                     .build_compute(Arc::clone(&device));
    
    let command_pool = CommandPool::new(Arc::clone(&device), compute_family);
    let command_buffers = command_pool.create_command_buffers(Arc::clone(&device), num_sets);
    
    let mut fences = Vec::with_capacity(num_sets as usize);
    for _ in 0..num_sets as usize {
      fences.push(Fence::new(Arc::clone(&device)));
    }
    
    Compute {
      queue: compute_queue,
      _family: compute_family,
      shader: compute_shader,
      command_pool,
      command_buffers,
      fences,
      descriptor_sets,
      pipeline,
    }
  }
  
  pub fn build(&mut self, device: Arc<Device>, graphics_queue: u32, buffers: Vec<&Buffer<f32>>) {
    let width = 256;
    let height = 256;
    for i in 0..self.command_buffers.len() {
      UpdateDescriptorSets::new()
      .add_storage_buffer(0, &buffers[i])
      .finish_update(Arc::clone(&device), &self.descriptor_sets[i]);
      
      let mut cmd = CommandBufferBuilder::primary_one_time_submit(Arc::clone(&self.command_buffers[i]));
      cmd = cmd.begin_command_buffer(Arc::clone(&device));
      
      //cmd = cmd.image_barrier(Arc::clone(&device), &Access::ColourAttachmentRead, &Access::ColourAttachmentWrite, &ImageLayout::ColourAttachmentOptimal, &ImageLayout::General, &ImageAspect::Colour, PipelineStage::FragmentShader, PipelineStage::ComputeShader, graphics_queue, self.queue as u32, image[i]);
      cmd = cmd.buffer_barrier(Arc::clone(&device), PipelineStage::ComputeShader, PipelineStage::ComputeShader, &Access::HostRead, &Access::MemoryWrite, graphics_queue, graphics_queue, &buffers[i]);
      
  //    let (width, height) = image[i].get_size();
      cmd = cmd.compute_dispatch(Arc::clone(&device), &self.pipeline, vec!(*self.descriptor_sets[i].set(0)), width / 16, height / 16, 1);
      
//      cmd = cmd.image_barrier(Arc::clone(&device), &Access::ColourAttachmentWrite, &Access::ColourAttachmentRead, &ImageLayout::General, &ImageLayout::ColourAttachmentOptimal, &ImageAspect::Colour, PipelineStage::ComputeShader, PipelineStage::FragmentShader, self.queue as u32, graphics_queue, image[i]);
     cmd = cmd.buffer_barrier(Arc::clone(&device), PipelineStage::ComputeShader, PipelineStage::ComputeShader, &Access::MemoryRead, &Access::HostWrite, graphics_queue, graphics_queue, &buffers[i]);
      
      cmd.end_command_buffer(Arc::clone(&device));
    }
  }
  
  pub fn destroy(&self, device: Arc<Device>) {
    for fence in &self.fences {
      fence.wait(Arc::clone(&device));
      fence.destroy(Arc::clone(&device));
    }
    
    self.shader.destroy(Arc::clone(&device));
    
    self.command_pool.destroy(Arc::clone(&device));
    
    self.pipeline.destroy(Arc::clone(&device));
    
    for descriptor in &self.descriptor_sets {
      descriptor.destroy(Arc::clone(&device));
    }
  }
}*/

































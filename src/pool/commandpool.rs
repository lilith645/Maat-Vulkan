use vk;

use crate::ownage::check_errors;

use crate::Device;
use crate::buffer::CommandBuffer;

use std::mem;
use std::ptr;
use std::sync::Arc;

pub struct CommandPool {
  pool: vk::CommandPool,
}

impl CommandPool {
  pub fn new(device: Arc<Device>, graphics_family: u32) -> CommandPool {
    let vk = device.pointers();
    let device = device.internal_object();
    
    let mut command_pool: vk::CommandPool = unsafe { mem::MaybeUninit::uninit().assume_init() };
    
    let command_pool_info = vk::CommandPoolCreateInfo {
      sType: vk::STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      pNext: ptr::null(),
      flags: vk::COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,//vk::COMMAND_POOL_CREATE_TRANSIENT_BIT, //to use vkResetCommandBuffer change to vk::COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
      queueFamilyIndex: graphics_family,
    };
    
    unsafe {
      check_errors(vk.CreateCommandPool(*device, &command_pool_info, ptr::null(), &mut command_pool));
    }
    
    CommandPool {
      pool: command_pool,
    }
  }
  
  pub fn new_transient(device: Arc<Device>, graphics_family: u32) -> CommandPool {
    let vk = device.pointers();
    let device = device.internal_object();
    
    let mut command_pool: vk::CommandPool = unsafe { mem::MaybeUninit::uninit().assume_init() };
    
    let command_pool_info = vk::CommandPoolCreateInfo {
      sType: vk::STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      pNext: ptr::null(),
      flags: vk::COMMAND_POOL_CREATE_TRANSIENT_BIT,
      queueFamilyIndex: graphics_family,
    };
    
    unsafe {
      check_errors(vk.CreateCommandPool(*device, &command_pool_info, ptr::null(), &mut command_pool));
    }
    
    CommandPool {
      pool: command_pool,
    }
  }
  
  pub fn local_command_pool(&self) -> &vk::CommandPool {
    &self.pool
  }
  
  pub fn create_command_buffers(&self, device: Arc<Device>, num_command_command_buffers: u32) -> Vec<Arc<CommandBuffer>> {
    let mut command_buffers: Vec<vk::CommandBuffer> = Vec::with_capacity(num_command_command_buffers as usize);
    
    let allocate_command_buffer_info = vk::CommandBufferAllocateInfo {
      sType: vk::STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      pNext: ptr::null(),
      commandPool: self.pool,
      level: vk::COMMAND_BUFFER_LEVEL_PRIMARY,
      commandBufferCount: num_command_command_buffers,
    };
    
    let vk = device.pointers();
    let device = device.internal_object();
    
    unsafe {
      check_errors(vk.AllocateCommandBuffers(*device, &allocate_command_buffer_info, command_buffers.as_mut_ptr()));
      command_buffers.set_len(num_command_command_buffers as usize);
    }
    
    command_buffers.iter().map(|x| Arc::new(CommandBuffer::from_buffer(*x))).collect::<Vec<Arc<CommandBuffer>>>()
  }
  
  pub fn destroy(&self, device: Arc<Device>) {
    let vk = device.pointers();
    let device = device.internal_object();
    
    println!("Destroying CommandPool");
    
    unsafe {
      vk.DestroyCommandPool(*device, self.pool, ptr::null());
    }
  }
}

use vk;

use crate::Device;
use crate::RenderPass;
use crate::ownage::check_errors;

use std::mem;
use std::ptr;
use std::sync::Arc;

pub struct Framebuffer {
  framebuffer: vk::Framebuffer,
}

impl Framebuffer {
  pub fn new(device: Arc<Device>, render_pass: &RenderPass, extent: &vk::Extent2D, image_view: &vk::ImageView) -> Framebuffer {
    let mut framebuffer: vk::Framebuffer = unsafe { mem::MaybeUninit::uninit().assume_init() };
    
    let framebuffer_create_info = vk::FramebufferCreateInfo {
      sType: vk::STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      pNext: ptr::null(),
      flags: 0,
      renderPass: *render_pass.internal_object(),
      attachmentCount: render_pass.get_num_attachments(),
      pAttachments: image_view,
      width: extent.width,
      height: extent.height,
      layers: 1,
    };
    
    let vk = device.pointers();
    let device = device.internal_object();
    
    unsafe {
      check_errors(vk.CreateFramebuffer(*device, &framebuffer_create_info, ptr::null(), &mut framebuffer));
    }
    
    Framebuffer {
      framebuffer,
    }
  }
  
  pub fn new_with_depth(device: Arc<Device>, render_pass: &RenderPass, extent: &vk::Extent2D, image_view: &vk::ImageView, depth_view: &vk::ImageView) -> Framebuffer {
    let mut framebuffer: vk::Framebuffer = unsafe { mem::MaybeUninit::uninit().assume_init() };
    
    let attachments = vec!(*image_view, *depth_view);
    
    let framebuffer_create_info = vk::FramebufferCreateInfo {
      sType: vk::STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      pNext: ptr::null(),
      flags: 0,
      renderPass: *render_pass.internal_object(),
      attachmentCount: render_pass.get_num_attachments(),
      pAttachments: attachments.as_ptr(),
      width: extent.width,
      height: extent.height,
      layers: 1,
    };
    
    let vk = device.pointers();
    let device = device.internal_object();
    
    unsafe {
      check_errors(vk.CreateFramebuffer(*device, &framebuffer_create_info, ptr::null(), &mut framebuffer));
    }
    
    Framebuffer {
      framebuffer,
    }
  }
  
  pub fn new_with_imageviews(device: Arc<Device>, render_pass: &RenderPass, extent: &vk::Extent2D, attachments: Vec<vk::ImageView>) -> Framebuffer {
    let mut framebuffer: vk::Framebuffer = unsafe { mem::MaybeUninit::uninit().assume_init() };
    
    let framebuffer_create_info = vk::FramebufferCreateInfo {
      sType: vk::STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      pNext: ptr::null(),
      flags: 0,
      renderPass: *render_pass.internal_object(),
      attachmentCount: render_pass.get_num_attachments(),
      pAttachments: attachments.as_ptr(),
      width: extent.width,
      height: extent.height,
      layers: 1,
    };
    
    let vk = device.pointers();
    let device = device.internal_object();
    
    unsafe {
      check_errors(vk.CreateFramebuffer(*device, &framebuffer_create_info, ptr::null(), &mut framebuffer));
    }
    
    Framebuffer {
      framebuffer,
    }
  }
  
  pub fn internal_object(&self) -> &vk::Framebuffer {
    &self.framebuffer
  }
  
  pub fn destroy(&self, device: Arc<Device>) {
    let vk = device.pointers();
    let device = device.internal_object();
    unsafe {
      vk.DestroyFramebuffer(*device, self.framebuffer, ptr::null());
    }
  }
}


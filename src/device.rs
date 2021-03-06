use vk;

use crate::Instance;
use crate::vkenums::{VkBool};

use crate::Logs;

use std::mem;
use std::ptr;
use std::sync::Arc;
use std::ffi::CStr;
use std::ffi::CString;

const PRIORITY: f32 = 0.5;

pub struct DeviceV2 {
  vk: vk::DevicePointers,
  device: vk::Device,
  phys_device: vk::PhysicalDevice,
  min_uniformbuffer_offset_alignment: u64,
  non_coherent_atom_size: u64,
}

impl DeviceV2 {
  pub fn new(instance: Arc<Instance>, surface: &vk::SurfaceKHR, debug: bool, logs: &mut Logs) -> Arc<DeviceV2> {
    let phys_device = DeviceV2::find_suitable_device(Arc::clone(&instance), logs);
    let (device, min_uniformbuffer_offset_alignment, non_coherent_atom_size) = DeviceV2::create_device(Arc::clone(&instance), phys_device, logs); 
    let vk = DeviceV2::create_device_instance(Arc::clone(&instance), &device);
    
    Arc::new(DeviceV2 {
      vk,
      device,
      phys_device,
      min_uniformbuffer_offset_alignment,
      non_coherent_atom_size,
    })
  }
  
  pub fn pointers(&self) -> &vk::DevicePointers {
    &self.vk
  }
  
  pub fn internal_object(&self) -> &vk::Device {
    &self.device
  }
  
  pub fn physical_device(&self) -> &vk::PhysicalDevice {
    &self.phys_device
  }
  
  pub fn get_min_uniformbuffer_offset_alignment(&self) -> u64 {
    self.min_uniformbuffer_offset_alignment
  }
  
  pub fn get_non_coherent_atom_size(&self) -> u64 {
    self.non_coherent_atom_size
  }
  
  pub fn get_device_queue(&self, family: u32, index: u32) -> vk::Queue {
    let mut graphics_queue: vk::Queue = unsafe { mem::MaybeUninit::uninit().assume_init() };
    
    unsafe {
      self.vk.GetDeviceQueue(self.device, family, index, &mut graphics_queue);
    }
    
    graphics_queue
  }
  
  pub fn physical_device_memory_properties(&self, instance: Arc<Instance>) -> vk::PhysicalDeviceMemoryProperties {
    let mut physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties = unsafe { mem::MaybeUninit::uninit().assume_init() };
    
    unsafe {
      let vk = instance.pointers();
      
      vk.GetPhysicalDeviceMemoryProperties(*self.physical_device(), &mut physical_device_memory_properties);
    }
    
    physical_device_memory_properties
  }
  
  pub fn wait(&self) {
    unsafe {
      self.vk.DeviceWaitIdle(self.device);
    }
  }
  
  pub fn destroy(&self) {
    unsafe {
      self.vk.DestroyDevice(self.device, ptr::null());
    }
  }
  
  fn create_device_instance(instance: Arc<Instance>, device: &vk::Device) -> vk::DevicePointers {
    let vk = instance.pointers();
    
    let vk_device = vk::DevicePointers::load(|name| unsafe {
      vk.GetDeviceProcAddr(*device, name.as_ptr()) as *const _
    });
    
    vk_device
  }
  
  fn is_device_suitable(instance: Arc<Instance>, phys_device: vk::PhysicalDevice) -> u32 {
    let features = instance.get_device_features(&phys_device);
    let properties = instance.get_device_properties(&phys_device);
    let queue_family_properties = instance.get_device_queue_family_properties(&phys_device);
    
    let mut score = 0;
    
    if properties.deviceType == vk::PHYSICAL_DEVICE_TYPE_DISCRETE_GPU {
      score += 1000;
    }
    
    for queue_family in &queue_family_properties {
      if queue_family.queueFlags & vk::QUEUE_GRAPHICS_BIT == 1 {
        score += 500;
      }
      if queue_family.queueFlags & vk::QUEUE_COMPUTE_BIT == 1 {
        score += 200;
      }
      if queue_family.queueFlags & vk::QUEUE_TRANSFER_BIT == 1 {
        score += 100;
      }
    }
    
    score += properties.limits.maxImageDimension2D;
    
    if features.geometryShader == 0 {
      return 0;
    }
    
    score
  }
  
  fn find_suitable_device(instance: Arc<Instance>, log: &mut Logs) -> vk::PhysicalDevice {
    let phys_devices = instance.enumerate_physical_devices(log);
    
    let mut scored_devices = Vec::new();
    
    for i in 0..phys_devices.len() {
      let score = DeviceV2::is_device_suitable(Arc::clone(&instance), phys_devices[i]);
      scored_devices.push((score, phys_devices[i]));
    }
    
    scored_devices.sort_by(|a, b| a.0.cmp(&b.0));
    
    scored_devices[0].1
  }
  
  fn create_device(instance: Arc<Instance>, phys_device: vk::PhysicalDevice, logs: &mut Logs) -> (vk::Device, u64, u64) {
    let mut device: vk::Device = unsafe { mem::MaybeUninit::uninit().assume_init() };
    
    let mut device_available_extensions = Vec::new();
    
    let device_extensions = instance.enumerate_device_extension_properties(&phys_device);
    let family_properties = instance.get_device_queue_family_properties(&phys_device);
    
    let layer_names = instance.get_layers();
    let layers_names_raw: Vec<*const i8> = layer_names.iter().map(|raw_name| raw_name.as_ptr()).collect();
    
    let mut available_extensions = instance.get_extensions();
    available_extensions.push(CString::new("VK_KHR_swapchain").unwrap());
    available_extensions.push(CString::new("VK_KHR_display_swapchain").unwrap());
    
    if true {
      available_extensions.push(CString::new("VK_EXT_debug_markers").unwrap());
    }
    
    let supported_device_extensions: Vec<CString>
       = device_extensions.iter().map(|x| unsafe { CStr::from_ptr(x.extensionName.as_ptr()) }.to_owned()).collect();
      logs.system_msg(&format!("\nSupported extensions:\n {:?}", supported_device_extensions));
      for supported_device_extension in supported_device_extensions {
        for available_extension in &available_extensions {
          if *available_extension == supported_device_extension {
            device_available_extensions.push(supported_device_extension.clone());
          }
        }
      }
    logs.system_msg(&format!("\nAvailable extensions:\n {:?}", device_available_extensions));
    let device_available_extensions_raw: Vec<*const i8> = device_available_extensions.iter().map(|raw_name| raw_name.as_ptr()).collect();
    
    let mut device_queue_infos = Vec::with_capacity(family_properties.len());
    
    for j in 0..family_properties.len() {
      const PRIORITY: f32 = 0.5;
      device_queue_infos.push(
        vk::DeviceQueueCreateInfo {
          sType: vk::STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
          pNext: ptr::null(),
          flags: 0,
          queueFamilyIndex: j as u32,
          queueCount: family_properties[j].queueCount,
          pQueuePriorities: &(PRIORITY.clone()),
        }
      );
    }
        
    let device_features: vk::PhysicalDeviceFeatures = instance.get_device_features(&phys_device);
    
    let features = vk::PhysicalDeviceFeatures {
      robustBufferAccess: VkBool::False.to_bits(),
      fullDrawIndexUint32: VkBool::False.to_bits(),
      imageCubeArray: VkBool::False.to_bits(),
      independentBlend: VkBool::False.to_bits(),
      geometryShader: VkBool::False.to_bits(),
      tessellationShader: VkBool::False.to_bits(),
      sampleRateShading: VkBool::False.to_bits(),
      dualSrcBlend: VkBool::False.to_bits(),
      logicOp: VkBool::False.to_bits(),
      multiDrawIndirect: VkBool::False.to_bits(),
      drawIndirectFirstInstance: VkBool::False.to_bits(),
      depthClamp: VkBool::False.to_bits(),
      depthBiasClamp: VkBool::False.to_bits(),
      fillModeNonSolid: VkBool::False.to_bits(),
      depthBounds: VkBool::False.to_bits(),
      wideLines: VkBool::False.to_bits(),
      largePoints: VkBool::False.to_bits(),
      alphaToOne: VkBool::False.to_bits(),
      multiViewport: VkBool::False.to_bits(),
      samplerAnisotropy: VkBool::False.to_bits(),
      textureCompressionETC2: VkBool::False.to_bits(),
      textureCompressionASTC_LDR: VkBool::False.to_bits(),
      textureCompressionBC: VkBool::False.to_bits(),
      occlusionQueryPrecise: VkBool::False.to_bits(),
      pipelineStatisticsQuery: VkBool::False.to_bits(),
      vertexPipelineStoresAndAtomics: VkBool::False.to_bits(),
      fragmentStoresAndAtomics: VkBool::False.to_bits(),
      shaderTessellationAndGeometryPointSize: VkBool::False.to_bits(),
      shaderImageGatherExtended: VkBool::False.to_bits(),
      shaderStorageImageExtendedFormats: VkBool::False.to_bits(),
      shaderStorageImageMultisample: VkBool::False.to_bits(),
      shaderStorageImageReadWithoutFormat: VkBool::False.to_bits(),
      shaderStorageImageWriteWithoutFormat: VkBool::False.to_bits(),
      shaderUniformBufferArrayDynamicIndexing: VkBool::False.to_bits(),
      shaderSampledImageArrayDynamicIndexing: VkBool::False.to_bits(),
      shaderStorageBufferArrayDynamicIndexing: VkBool::False.to_bits(),
      shaderStorageImageArrayDynamicIndexing: VkBool::False.to_bits(),
      shaderClipDistance: VkBool::False.to_bits(),
      shaderCullDistance: VkBool::False.to_bits(),
      shaderf3264: VkBool::False.to_bits(),
      shaderInt64: VkBool::False.to_bits(),
      shaderInt16: VkBool::False.to_bits(),
      shaderResourceResidency: VkBool::False.to_bits(),
      shaderResourceMinLod: VkBool::False.to_bits(),
      sparseBinding: VkBool::False.to_bits(),
      sparseResidencyBuffer: VkBool::False.to_bits(),
      sparseResidencyImage2D: VkBool::False.to_bits(),
      sparseResidencyImage3D: VkBool::False.to_bits(),
      sparseResidency2Samples: VkBool::False.to_bits(),
      sparseResidency4Samples: VkBool::False.to_bits(),
      sparseResidency8Samples: VkBool::False.to_bits(),
      sparseResidency16Samples: VkBool::False.to_bits(),
      sparseResidencyAliased: VkBool::False.to_bits(),
      variableMultisampleRate: VkBool::False.to_bits(),
      inheritedQueries: VkBool::False.to_bits(),
    };
    
    let device_info = vk::DeviceCreateInfo {
      sType: vk::STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      pNext: ptr::null(),
      flags: 0,
      queueCreateInfoCount: family_properties.len() as u32,
      pQueueCreateInfos: device_queue_infos.as_ptr(),
      ppEnabledLayerNames: layers_names_raw.as_ptr(),
      enabledLayerCount: layers_names_raw.len() as u32,
      ppEnabledExtensionNames: device_available_extensions_raw.as_ptr(),
      enabledExtensionCount: device_available_extensions_raw.len() as u32,
      pEnabledFeatures: &features,
    };
    
    device = instance.create_device(&phys_device, &device_info, logs);
    
    let properties = instance.get_device_properties(&phys_device);
    let min_uniformbuffer_offset_alignment = properties.limits.minUniformBufferOffsetAlignment;
    let non_coherent_atom_size = properties.limits.nonCoherentAtomSize;
    
    (device, min_uniformbuffer_offset_alignment, non_coherent_atom_size)
  }
}




































pub struct Device {
  vk: vk::DevicePointers,
  device: vk::Device,
  phys_device: vk::PhysicalDevice,
  
  min_uniformbuffer_offset_alignment: u64,
  non_coherent_atom_size: u64,
  _extensions: Vec<CString>,
}

impl Device {
  pub fn new(instance: Arc<Instance>, surface: &vk::SurfaceKHR, debug: bool, logs: &mut Logs) -> Arc<Device> {
    let (device, phys_device, min_uniformbuffer_offset_alignment, non_coherent_atom_size, extensions) = Device::create_suitable_device(Arc::clone(&instance), surface, debug, logs);
    let vk = Device::create_device_instance(Arc::clone(&instance), &device);
    
    Arc::new(Device {
      vk,
      device,
      phys_device,
      min_uniformbuffer_offset_alignment,
      non_coherent_atom_size,
      _extensions: extensions,
    })
  }
  
  pub fn pointers(&self) -> &vk::DevicePointers {
    &self.vk
  }
  
  pub fn internal_object(&self) -> &vk::Device {
    &self.device
  }
  
  pub fn get_min_uniformbuffer_offset_alignment(&self) -> u64 {
    self.min_uniformbuffer_offset_alignment
  }
  
  pub fn get_non_coherent_atom_size(&self) -> u64 {
    self.non_coherent_atom_size
  }
  
  pub fn min_buffer_align(&self, _buffer: &vk::Buffer) -> u64 {
//    let mem_req: Vec<> = unsafe { mem::MaybeUninit::uninit().assume_init() }; 
  //  self.vk.GetBufferMemoryRequirements(self.device, *buffer, mem_req.as_mut_ptr());
    
   // mem_req.size
   0
  }
  
  pub fn physical_device(&self) -> &vk::PhysicalDevice {
    &self.phys_device
  }
  
  pub fn get_device_queue(&self, family: u32, index: u32) -> vk::Queue {
    let mut graphics_queue: vk::Queue = unsafe { mem::MaybeUninit::uninit().assume_init() };
    
    unsafe {
      self.vk.GetDeviceQueue(self.device, family, index, &mut graphics_queue);
    }
    
    graphics_queue
  }
  
  pub fn physical_device_memory_properties(&self, instance: Arc<Instance>) -> vk::PhysicalDeviceMemoryProperties {
    let mut physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties = unsafe { mem::MaybeUninit::uninit().assume_init() };
    
    unsafe {
      let vk = instance.pointers();
      
      vk.GetPhysicalDeviceMemoryProperties(*self.physical_device(), &mut physical_device_memory_properties);
    }
    
    physical_device_memory_properties
  }
  
  pub fn best_compute_queue_nph(&self, instance: Arc<Instance>, queue_family_idx: &mut u32) -> vk::Result {
    let mut queue_family_properties_count = 0;
    println!("Start of best compute queue");
    unsafe {
      let vk = instance.pointers();
      
      vk.GetPhysicalDeviceQueueFamilyProperties(*self.physical_device(), &mut queue_family_properties_count, ptr::null_mut());
    }
    println!("queue count: {}", queue_family_properties_count);
    let mut queue_family_properties: Vec<vk::QueueFamilyProperties> = Vec::with_capacity(queue_family_properties_count as usize);//unsafe { mem::MaybeUninit::uninit().assume_init() };
    
    unsafe {
      let vk = instance.pointers();
      
      vk.GetPhysicalDeviceQueueFamilyProperties(*self.physical_device(), &mut queue_family_properties_count, queue_family_properties.as_mut_ptr());
      
      queue_family_properties.set_len(queue_family_properties_count as usize);
    }
    
    println!("Queue family properties");
    for i in 0..queue_family_properties_count as usize {
      let masked_flags = !(vk::QUEUE_TRANSFER_BIT | vk::QUEUE_SPARSE_BINDING_BIT) & queue_family_properties[i].queueFlags;
      
      if (!(vk::QUEUE_GRAPHICS_BIT & masked_flags) != 0 && (vk::QUEUE_COMPUTE_BIT & masked_flags) != 0) {
        *queue_family_idx = i as u32;
        return vk::SUCCESS;
      }
    }
    
    for i in 0..queue_family_properties_count as usize {
      let masked_flags = (!(vk::QUEUE_TRANSFER_BIT | vk::QUEUE_SPARSE_BINDING_BIT) & queue_family_properties[i].queueFlags);
      
      if (vk::QUEUE_COMPUTE_BIT & masked_flags) != 0 {
        *queue_family_idx = i as u32;
        return vk::SUCCESS;
      }
    }
    
    vk::ERROR_INITIALIZATION_FAILED
  }
  
  pub fn best_transfer_queue_nph(&self, instance: Arc<Instance>, queue_family_idx: &mut u32) -> vk::Result {
    /*let mut queue_family_properties_count = 0;
    
    unsafe {
      let vk = instance.pointers();
      
      vk.GetPhysicalDeviceQueueFamilyProperties(*self.physical_device(), &mut queue_family_properties_count, ptr::null_mut());
    }
    
    let mut queue_family_properties: Vec<vk::QueueFamilyProperties> = unsafe { mem::MaybeUninit::uninit().assume_init() };
    
    unsafe {
      let vk = instance.pointers();
      
      vk.GetPhysicalDeviceQueueFamilyProperties(*self.physical_device(), &mut queue_family_properties_count, queue_family_properties.as_mut_ptr());
    }
    
    for i in 0..queue_family_properties_count as usize {
      let masked_flags = (!vk::QUEUE_SPARSE_BINDING_BIT & queue_family_properties[i].queueFlags);
      
      if (!((vk::QUEUE_GRAPHICS_BIT | vk::QUEUE_COMPUTE_BIT) & masked_flags) != 0 &&
          (vk::QUEUE_TRANSFER_BIT & masked_flags) != 0) {
        *queue_family_idx = i as u32;
        return vk::SUCCESS;
      }
    }
    
    for i in 0..queue_family_properties_count as usize {
      let masked_flags = (!vk::QUEUE_SPARSE_BINDING_BIT & queue_family_properties[i].queueFlags);
      
      if (!(vk::QUEUE_GRAPHICS_BIT & masked_flags) != 0 && (vk::QUEUE_COMPUTE_BIT & masked_flags) != 0) {
        *queue_family_idx = i as u32;
        return vk::SUCCESS;
      }
    }
    
    for i in 0..queue_family_properties_count as usize {
      let masked_flags = (!vk::QUEUE_SPARSE_BINDING_BIT & queue_family_properties[i].queueFlags);
      
      if ((vk::QUEUE_GRAPHICS_BIT | vk::QUEUE_COMPUTE_BIT | vk::QUEUE_TRANSFER_BIT) & masked_flags) != 0 {
        *queue_family_idx = i as u32;
        return vk::SUCCESS;
      }
    }
    */
    vk::ERROR_INITIALIZATION_FAILED
  }
  
  pub fn get_compute_queue(&self, instance: Arc<Instance>, logs: &mut Logs) -> (vk::Queue, u32) {
    let mut num_queue_families = 0;
    let mut queue_family_properties: Vec<vk::QueueFamilyProperties>;
    let mut compute_index: u32 = 0;
    let mut compute_queue = 0;
    
    let vk_instance = instance.pointers();
    
    unsafe {
      vk_instance.GetPhysicalDeviceQueueFamilyProperties(self.phys_device, &mut num_queue_families, ptr::null_mut());
      
      queue_family_properties = Vec::with_capacity(num_queue_families as usize);
      
      vk_instance.GetPhysicalDeviceQueueFamilyProperties(self.phys_device, &mut num_queue_families, queue_family_properties.as_mut_ptr());
      queue_family_properties.set_len(num_queue_families as usize);
    }
    
    for i in 0..num_queue_families as usize {
      if Device::has_compute_bit(&queue_family_properties[i].queueFlags) && !Device::has_graphics_bit(&queue_family_properties[i].queueFlags) {
        compute_index = i as u32;
        logs.system_msg(&format!("Dedicated Compute queue found!"));
        break;
      }
    }
    
    unsafe {
      self.vk.GetDeviceQueue(self.device, compute_index, 0, &mut compute_queue);
    }
    
    (compute_queue, compute_index)
  }
  
  pub fn wait(&self) {
    unsafe {
   //   println!("Waiting for device to idle");
      self.vk.DeviceWaitIdle(self.device);
    }
  }
  
  pub fn destroy(&self) {
    unsafe {
      self.vk.DestroyDevice(self.device, ptr::null());
    }
  }
  
  fn create_device_instance(instance: Arc<Instance>, device: &vk::Device) -> vk::DevicePointers {
    let vk = instance.pointers();
    
    let vk_device = vk::DevicePointers::load(|name| unsafe {
      vk.GetDeviceProcAddr(*device, name.as_ptr()) as *const _
    });
    
    vk_device
  }
  
  fn create_suitable_device(instance: Arc<Instance>, surface: &vk::SurfaceKHR, debug: bool, logs: &mut Logs) -> (vk::Device, vk::PhysicalDevice, u64, u64, Vec<CString>) {
    let layer_names = instance.get_layers();
    logs.system_msg(&format!("\nLayer names: {:?}", layer_names));
    let layers_names_raw: Vec<*const i8> = layer_names.iter().map(|raw_name| raw_name.as_ptr()).collect();
    
    let physical_devices = instance.enumerate_physical_devices(logs);
    
    Device::print_physical_device_details(instance.pointers(), &physical_devices, logs);
    
    let mut device: vk::Device = unsafe { mem::MaybeUninit::uninit().assume_init() };
    let mut device_available_extensions = Vec::new();
    let mut physical_device_index = 0;
    
    for i in 0..physical_devices.len() {
      logs.system_msg(&format!("\nLooping devices: {}", i));
      let family_properties = instance.get_device_queue_family_properties(&physical_devices[i]);
      
      let mut has_graphics_bit = false;
      let mut device_supports_surface: u32 = 0;
      
      for j in 0..family_properties.len() {
        let queue_flags = family_properties[j].queueFlags;
        if Device::has_graphics_bit(&queue_flags) {
          has_graphics_bit = true;
        }
        
        if device_supports_surface == 0 {
          device_supports_surface = instance.physical_device_supports_surface(&physical_devices[i], j as u32, surface);
        }
      }
      
      if has_graphics_bit && device_supports_surface != 0 {
        let device_extensions = instance.enumerate_device_extension_properties(&physical_devices[i]);
        
        let mut available_extensions = instance.get_extensions();
        available_extensions.push(CString::new("VK_KHR_swapchain").unwrap());
        available_extensions.push(CString::new("VK_KHR_display_swapchain").unwrap());
        //available_extensions.push(CString::new("VK_KHR_sampler_mirror_clamp_to_edge").unwrap());
        // available_extensions.push(CString::new("VK_KHR_get_memory_requirements2").unwrap());
        //available_extensions.push(CString::new("VK_KHR_dedicated_allocation").unwrap());
        //available_extensions.push(CString::new("VK_KHR_incremental_present").unwrap());
        
        if debug {
          available_extensions.push(CString::new("VK_EXT_debug_markers").unwrap());
        }
        
        let supported_device_extensions: Vec<CString>
           = device_extensions.iter().map(|x| unsafe { CStr::from_ptr(x.extensionName.as_ptr()) }.to_owned()).collect();
          logs.system_msg(&format!("\nSupported extensions:\n {:?}", supported_device_extensions));
          for supported_device_extension in supported_device_extensions {
            for available_extension in &available_extensions {
              if *available_extension == supported_device_extension {
                device_available_extensions.push(supported_device_extension.clone());
              }
            }
          }
        logs.system_msg(&format!("\nAvailable extensions:\n {:?}", device_available_extensions));
        let device_available_extensions_raw: Vec<*const i8> = device_available_extensions.iter().map(|raw_name| raw_name.as_ptr()).collect();
        
        let mut device_queue_infos = Vec::with_capacity(family_properties.len());
        
        for j in 0..family_properties.len() {
          const PRIORITY: f32 = 0.5;
          device_queue_infos.push(
            vk::DeviceQueueCreateInfo {
              sType: vk::STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
              pNext: ptr::null(),
              flags: 0,
              queueFamilyIndex: j as u32,
              queueCount: family_properties[j].queueCount,
              pQueuePriorities: &(PRIORITY.clone()),
            }
          );
        }
        
        let device_features: vk::PhysicalDeviceFeatures = instance.get_device_features(&physical_devices[physical_device_index]);
        
        match device_features.shaderSampledImageArrayDynamicIndexing {
          vk::TRUE => {
            logs.system_msg(&format!("Dynamic indexing supported!"));
          },
          _ => {logs.warning_msg(&format!("Dynamic indexing not supported :("));}
        }
        
        logs.system_msg(&format!("feature alpha to one {}", device_features.alphaToOne));
        
        // Need to fix
        let features = vk::PhysicalDeviceFeatures {
          robustBufferAccess: VkBool::False.to_bits(),
          fullDrawIndexUint32: VkBool::False.to_bits(),
          imageCubeArray: VkBool::False.to_bits(),
          independentBlend: VkBool::False.to_bits(),
          geometryShader: VkBool::False.to_bits(),
          tessellationShader: VkBool::False.to_bits(),
          sampleRateShading: VkBool::False.to_bits(),
          dualSrcBlend: VkBool::False.to_bits(),
          logicOp: VkBool::False.to_bits(),
          multiDrawIndirect: VkBool::False.to_bits(),
          drawIndirectFirstInstance: VkBool::False.to_bits(),
          depthClamp: VkBool::False.to_bits(),
          depthBiasClamp: VkBool::False.to_bits(),
          fillModeNonSolid: VkBool::False.to_bits(),
          depthBounds: VkBool::False.to_bits(),
          wideLines: VkBool::False.to_bits(),
          largePoints: VkBool::False.to_bits(),
          alphaToOne: VkBool::False.to_bits(),
          multiViewport: VkBool::False.to_bits(),
          samplerAnisotropy: VkBool::False.to_bits(),
          textureCompressionETC2: VkBool::False.to_bits(),
          textureCompressionASTC_LDR: VkBool::False.to_bits(),
          textureCompressionBC: VkBool::False.to_bits(),
          occlusionQueryPrecise: VkBool::False.to_bits(),
          pipelineStatisticsQuery: VkBool::False.to_bits(),
          vertexPipelineStoresAndAtomics: VkBool::False.to_bits(),
          fragmentStoresAndAtomics: VkBool::False.to_bits(),
          shaderTessellationAndGeometryPointSize: VkBool::False.to_bits(),
          shaderImageGatherExtended: VkBool::False.to_bits(),
          shaderStorageImageExtendedFormats: VkBool::False.to_bits(),
          shaderStorageImageMultisample: VkBool::False.to_bits(),
          shaderStorageImageReadWithoutFormat: VkBool::False.to_bits(),
          shaderStorageImageWriteWithoutFormat: VkBool::False.to_bits(),
          shaderUniformBufferArrayDynamicIndexing: VkBool::False.to_bits(),
          shaderSampledImageArrayDynamicIndexing: VkBool::False.to_bits(),
          shaderStorageBufferArrayDynamicIndexing: VkBool::False.to_bits(),
          shaderStorageImageArrayDynamicIndexing: VkBool::False.to_bits(),
          shaderClipDistance: VkBool::False.to_bits(),
          shaderCullDistance: VkBool::False.to_bits(),
          shaderf3264: VkBool::False.to_bits(),
          shaderInt64: VkBool::False.to_bits(),
          shaderInt16: VkBool::False.to_bits(),
          shaderResourceResidency: VkBool::False.to_bits(),
          shaderResourceMinLod: VkBool::False.to_bits(),
          sparseBinding: VkBool::False.to_bits(),
          sparseResidencyBuffer: VkBool::False.to_bits(),
          sparseResidencyImage2D: VkBool::False.to_bits(),
          sparseResidencyImage3D: VkBool::False.to_bits(),
          sparseResidency2Samples: VkBool::False.to_bits(),
          sparseResidency4Samples: VkBool::False.to_bits(),
          sparseResidency8Samples: VkBool::False.to_bits(),
          sparseResidency16Samples: VkBool::False.to_bits(),
          sparseResidencyAliased: VkBool::False.to_bits(),
          variableMultisampleRate: VkBool::False.to_bits(),
          inheritedQueries: VkBool::False.to_bits(),
        };
        
        //features.robustBufferAccess = vk::TRUE;
        let device_info = vk::DeviceCreateInfo {
          sType: vk::STRUCTURE_TYPE_DEVICE_CREATE_INFO,
          pNext: ptr::null(),
          flags: 0,
          queueCreateInfoCount: family_properties.len() as u32,
          pQueueCreateInfos: device_queue_infos.as_ptr(),
          ppEnabledLayerNames: layers_names_raw.as_ptr(),
          enabledLayerCount: layers_names_raw.len() as u32,
          ppEnabledExtensionNames: device_available_extensions_raw.as_ptr(),
          enabledExtensionCount: device_available_extensions_raw.len() as u32,
          pEnabledFeatures: &features, // For more features use vk::GetPhysicalDeviceFeatures
        };
        
        logs.system_msg(&format!("Physical device index = {}", i));
        device = instance.create_device(&physical_devices[i], &device_info, logs);
        
        physical_device_index = i;
        break;
      }
    }
    
    
    let mut device_prop: vk::PhysicalDeviceProperties = unsafe { mem::MaybeUninit::uninit().assume_init() };
    
    unsafe {
      instance.pointers().GetPhysicalDeviceProperties(physical_devices[physical_device_index], &mut device_prop);
    }
    
    let min_uniformbuffer_offset_alignment = device_prop.limits.minUniformBufferOffsetAlignment;
    let non_coherent_atom_size = device_prop.limits.nonCoherentAtomSize;
    logs.system_msg(&format!("Max fragment shader outputs: {}", device_prop.limits.maxFragmentOutputAttachments));
    logs.system_msg(&format!("Max fragment shader inputs: {}", device_prop.limits.maxDescriptorSetInputAttachments));
    
    (device, physical_devices[physical_device_index], min_uniformbuffer_offset_alignment, non_coherent_atom_size, device_available_extensions)
  }
  
  fn print_physical_device_details(vk_instance: &vk::InstancePointers, physical_devices: &Vec<vk::PhysicalDevice>, logs: &mut Logs) {
    for i in 0..physical_devices.len() as usize {
      let mut device_prop: vk::PhysicalDeviceProperties = unsafe { mem::MaybeUninit::uninit().assume_init() };
      
      unsafe {
        vk_instance.GetPhysicalDeviceProperties(physical_devices[i], &mut device_prop);
      }
      
      logs.system_msg(&format!("min alignment: {}", device_prop.limits.minUniformBufferOffsetAlignment));
      logs.system_msg(&format!("max push constant size: {}", device_prop.limits.maxPushConstantsSize));
      let device_name = device_prop.deviceName.iter().map(|a| { 
        let mut b = (*a as u8 as char).to_string();
        if b == "\u{0}".to_string() {
          b = "".to_string();
        }
        b
      }).collect::<String>();
      
      let device_type = device_prop.deviceType;
      let mut device_type_name = "";
      
      match device_type {
       vk::PHYSICAL_DEVICE_TYPE_OTHER => { device_type_name = "Other GPU"; },
       vk::PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => { device_type_name = "Integrated GPU"; },
       vk::PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => { device_type_name = "Discrete GPU"; },
       vk::PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => { device_type_name = "Virtual GPU"; },
       vk::PHYSICAL_DEVICE_TYPE_CPU => { device_type_name = "CPU"; },
        _ => {},
      }
      
      logs.system_msg(&format!("{}: {} -> {}", i, device_type_name, device_name));
    }
    
    for i in 0..physical_devices.len() {
      logs.system_msg(&format!("Device: {}", i));
      let mut family_count = 0;
      
      unsafe {
        vk_instance.GetPhysicalDeviceQueueFamilyProperties(physical_devices[i], &mut family_count, ptr::null_mut());
      }
      
      let mut family_properties = Vec::with_capacity(family_count as usize);
      
      unsafe {
        vk_instance.GetPhysicalDeviceQueueFamilyProperties(physical_devices[i], &mut family_count, family_properties.as_mut_ptr());
        family_properties.set_len(family_count as usize);
      }
      
      //let mut queue_index = 0;
      for j in 0..family_properties.len() {
        logs.system_msg(&format!("  Queue: {}", j));
        let mut queue_flags = family_properties[j].queueFlags;
        if Device::has_graphics_bit(&queue_flags) {
          logs.system_msg(&format!("    Graphics: True"));
          queue_flags -= 1;
        } else {
          logs.system_msg(&format!("    Graphics: False"));
        };
        if queue_flags >= 8 {
          logs.system_msg(&format!("     Binding: True"));
          queue_flags -= 8;
        } else {
          logs.system_msg(&format!("     Binding: False"));
        }
        if queue_flags >= 4 {
          logs.system_msg(&format!("    Transfer: True"));
          queue_flags -= 4;
        } else {
          logs.system_msg(&format!("    Transfer: False"));
        }
        if queue_flags != 0 {
        
          logs.system_msg(&format!("     Compute: True"));
        } else {
          logs.system_msg(&format!("     Compute: False"));
        }
      }
    }
  }
  
  fn has_graphics_bit(queue_flags: &u32) -> bool {
    queue_flags % 2 != 0 
  }
  
  fn has_compute_bit(queue_flags: &u32) -> bool {
    let mut queue = *queue_flags;
    if Device::has_graphics_bit(queue_flags) {
      queue -= 1;
    }
    if queue >= 8 {
      queue -= 8;
    }
    if queue >= 4 {
      queue -= 4;
    }
    
    (queue != 0)
  }
}

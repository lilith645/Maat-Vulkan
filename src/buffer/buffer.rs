use vk;

use crate::buffer::{CommandBuffer, BufferUsage};
use crate::pool::CommandPool;
use crate::Instance;
use crate::Device;
use crate::ownage::check_errors;

use libc::memcpy;

use std::mem;
use std::ptr;
use std::sync::Arc;
use std::os::raw::c_void;

use crate::Logs;

#[derive(Clone)]
pub struct Buffer<T: Clone> {
  buffer: Vec<vk::Buffer>,
  memory: Vec<vk::DeviceMemory>,
  usage: BufferUsage,
  size: u64,
  data: Vec<T>,
}

impl<T: Clone> Buffer<T> {
  fn illegal_size(max_size: &u64, data: &Vec<T>) -> bool {
    *max_size < data.len() as u64*mem::size_of::<T>() as u64
  }
  
  fn align_data(device: Arc<Device>, data: &mut Vec<T>) {
    let mut buffer_size = mem::size_of::<T>() * data.len();
    let min_alignment = device.get_non_coherent_atom_size();
    
    while buffer_size as u64%min_alignment != 0 {
      let temp = data[data.len()-1].clone();
      data.push(temp);
      buffer_size = mem::size_of::<T>() * data.len();
    }
  }
  
  fn align_phantom_data(device: Arc<Device>, data_len: &mut u64) {
    let mut buffer_size = mem::size_of::<T>() as u64 * *data_len;
    let min_alignment = device.get_non_coherent_atom_size();
    
    while buffer_size as u64%min_alignment != 0 {
      *data_len += 1;
      buffer_size = mem::size_of::<T>() as u64 * *data_len;
    }
  }
  
  pub fn from_raw_buffer(buffer: vk::Buffer, memory: vk::DeviceMemory, usage: BufferUsage, size: u64, data: Vec<T>) -> Buffer<T> {
    Buffer {
      buffer: vec!(buffer),
      memory: vec!(memory),
      usage,
      size,
      data,
    }
  }
  
  pub fn compute_test(instance: Arc<Instance>, device: Arc<Device>, logs: &mut Logs) -> (Buffer<f32>, Buffer<f32>) {//(vk::Buffer, vk::Buffer, vk::DeviceMemory, u64, Vec<f32>, u32, u64) {
    //let properties: vk::MemoryPropertyFlags = vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT; //| vk::MEMORY_PROPERTY_HOST_COHERENT_BIT;
    println!("Start of compute test function");
    let properties = device.physical_device_memory_properties(Arc::clone(&instance));//instance.get_device_properties(device.physical_device());
    println!("before best compute queue");
    let mut queue_family_idx = 0;
    check_errors(device.best_compute_queue_nph(Arc::clone(&instance), &mut queue_family_idx));
    println!("After best compute queue");
   // let mut buffer: vk::Buffer = unsafe { mem::MaybeUninit::uninit().assume_init() };
   // let mut buffer_memory: vk::DeviceMemory = unsafe { mem::MaybeUninit::uninit().assume_init() };
    
    let mut in_data = Vec::new();
    let mut out_data = Vec::new();
    
    let buffer_length = 64;
    let buffer_size = (buffer_length * mem::size_of::<f32>()) as u64;
    for i in 0..buffer_length {
      in_data.push(0.5);
      out_data.push(0.0);
    }
    
    let memory_size = buffer_size*2;
    
    let in_buf = Buffer::cpu_buffer_with_data(Arc::clone(&instance), Arc::clone(&device), &BufferUsage::storage_buffer(), 1, in_data);
    let out_buf = Buffer::cpu_buffer_with_data(Arc::clone(&instance), Arc::clone(&device), &BufferUsage::storage_buffer(), 1, out_data);
    
    (in_buf, out_buf)
    /*
    let mut memory_type_index = vk::MAX_MEMORY_TYPES;
    
    let mut data = Vec::with_capacity(buffer_length);
    println!("before memory type index");
    for k in 0..properties.memoryTypeCount as usize {
      if (vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT & properties.memoryTypes[k].propertyFlags) != 0 &&
        (vk::MEMORY_PROPERTY_HOST_COHERENT_BIT & properties.memoryTypes[k].propertyFlags) != 0 &&
        (memory_size < properties.memoryHeaps[properties.memoryTypes[k].heapIndex as usize].size) {
        memory_type_index = k as u32;
        break;
      }
    }
    
    let memory_allocate_info = {
      vk::MemoryAllocateInfo {
        sType: vk::STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        pNext: ptr::null(),
        allocationSize: memory_size,
        memoryTypeIndex: memory_type_index as u32,
      }
    };
    println!("memory allocate info");
    unsafe {
      let vk = device.pointers();
      let device = device.internal_object();
      check_errors(vk.AllocateMemory(*device, &memory_allocate_info, ptr::null(), &mut buffer_memory));
      check_errors(vk.MapMemory(*device, buffer_memory, 0, memory_size, 0, data.as_mut_ptr() as *mut *mut c_void));
    }
    println!("allocated memory and mapped memory");
    for i in 0..buffer_length {
      data[i] = 0.5;
    }
    
    unsafe {
      let vk = device.pointers();
      let device = device.internal_object();
      
      vk.UnmapMemory(*device, buffer_memory);
    }
    
    let buffer_create_info = {
      vk::BufferCreateInfo {
        sType: vk::STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        pNext: ptr::null(),
        flags: 0,
        size: buffer_size as vk::DeviceSize,
        usage: BufferUsage::storage_buffer().to_bits(),
        sharingMode: vk::SHARING_MODE_EXCLUSIVE,
        queueFamilyIndexCount: 1,
        pQueueFamilyIndices: vec!(queue_family_idx).as_ptr(),
      }
    };
    println!("before buffer uninitialisation");
    let mut in_buffer: vk::Buffer = unsafe { mem::MaybeUninit::uninit().assume_init() };
    let mut out_buffer: vk::Buffer = unsafe { mem::MaybeUninit::uninit().assume_init() };
    println!("before buffer creation");
    unsafe {
      let vk = device.pointers();
      let device = device.internal_object();
      
      check_errors(vk.CreateBuffer(*device, &buffer_create_info, ptr::null(), &mut in_buffer));
      check_errors(vk.BindBufferMemory(*device, in_buffer, buffer_memory, 0));
      
      check_errors(vk.CreateBuffer(*device, &buffer_create_info, ptr::null(), &mut out_buffer));
      check_errors(vk.BindBufferMemory(*device, out_buffer, buffer_memory, buffer_size));
    }
    

    
    (in_buffer, out_buffer, buffer_memory, memory_size, data, queue_family_idx, buffer_size)*/
  }
  
  pub fn cpu_buffer(instance: Arc<Instance>, device: Arc<Device>, usage: BufferUsage, num_sets: u32, data_len: u64) -> Buffer<T> {
    let mut data_len = data_len;
    Buffer::<T>::align_phantom_data(Arc::clone(&device), &mut data_len);
    
    let mut buffers: Vec<vk::Buffer> = Vec::new();
    let mut memorys: Vec<vk::DeviceMemory> = Vec::new();
    
    for _ in 0..num_sets {
      let (buffer, memory) = Buffer::<T>::create_buffer(Arc::clone(&instance), Arc::clone(&device), &usage, vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT, data_len);
      buffers.push(buffer);
      memorys.push(memory);
    }
    
    let buffer = Buffer {
      buffer: buffers,
      memory: memorys,
      usage,
      size: mem::size_of::<T>() as u64*data_len,
      data: Vec::new(),
    };
    
    buffer
  }
  
  pub fn cpu_buffer_with_data(instance: Arc<Instance>, device: Arc<Device>, usage: &BufferUsage, num_sets: u32, data: Vec<T>) -> Buffer<T> {
    let mut data = data;
    Buffer::align_data(Arc::clone(&device), &mut data);
    let mut buffers: Vec<vk::Buffer> = Vec::new();
    let mut memorys: Vec<vk::DeviceMemory> = Vec::new();
    
    for _ in 0..num_sets {
      let (buffer, memory) = Buffer::<T>::create_buffer(Arc::clone(&instance), Arc::clone(&device), usage, vk::MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk::MEMORY_PROPERTY_HOST_COHERENT_BIT, data.len() as u64);
      buffers.push(buffer);
      memorys.push(memory);
    }
    
    let mut buffer = Buffer {
      buffer: buffers,
      memory: memorys,
      usage: usage.clone(),
      size: mem::size_of::<T>() as u64*data.len() as u64,
      data: data,
    };
    
    let data = buffer.internal_data();
    buffer.fill_entire_buffer_all_frames(Arc::clone(&device), data);
    
    buffer
  }
  
  pub fn device_local_buffer(instance: Arc<Instance>, device: Arc<Device>, usage: BufferUsage, num_sets: u32, data_len: u64) -> Buffer<T> {
    let mut data_len = data_len;
    Buffer::<T>::align_phantom_data(Arc::clone(&device), &mut data_len);
    let mut buffers: Vec<vk::Buffer> = Vec::new();
    let mut memorys: Vec<vk::DeviceMemory> = Vec::new();
    
    for _ in 0..num_sets {
      let (buffer, memory) = Buffer::<T>::create_buffer(Arc::clone(&instance), Arc::clone(&device), &usage, vk::MEMORY_PROPERTY_DEVICE_LOCAL_BIT, data_len);
      buffers.push(buffer);
      memorys.push(memory);
    }
    
    Buffer {
      buffer: buffers,
      memory: memorys,
      usage,
      size: mem::size_of::<T>() as u64*data_len,
      data: Vec::new(),
    }
  }
  
  pub fn device_local_buffer_with_data(instance: Arc<Instance>, device: Arc<Device>, command_pool: &CommandPool, graphics_queue: &vk::Queue, buffer_usage: BufferUsage, data: Vec<T>) -> Buffer<T> {
    let mut data = data;
    Buffer::align_data(Arc::clone(&device), &mut data);
    let mut buffer_usage = buffer_usage;
    buffer_usage.set_as_transfer_dst();
    
    let data_len = data.len() as u64;
    
    let usage_src = BufferUsage::transfer_src_buffer();
    let usage_dst = buffer_usage;
    
    let staging_buffer: Buffer<T> = Buffer::cpu_buffer_with_data(Arc::clone(&instance), Arc::clone(&device), &usage_src, 1, data);
    let buffer: Buffer<T> = Buffer::device_local_buffer(Arc::clone(&instance), Arc::clone(&device), usage_dst, 1, data_len);
    
    let command_buffer = CommandBuffer::begin_single_time_command(Arc::clone(&device), command_pool);
    command_buffer.copy_buffer(Arc::clone(&device), &staging_buffer, &buffer, 0);
    command_buffer.end_single_time_command(Arc::clone(&device), command_pool, graphics_queue);
    
    staging_buffer.destroy(Arc::clone(&device));
    
    buffer
  }
  
  pub fn _fill_partial_buffer(&mut self, device: Arc<Device>, current_buffer: usize, offset: u32, data: Vec<T>) {
    let mut data = data;
    Buffer::align_data(Arc::clone(&device), &mut data);
    if Buffer::illegal_size(&self.size, &data) {
      return;
    }
    self.data = data;
    
    let mut host_visible_data = unsafe { mem::MaybeUninit::uninit().assume_init() };
    let buffer_offset = mem::size_of::<T>() * offset as usize;
    let buffer_size = mem::size_of::<T>() * self.data.len();
    
    unsafe {
      let vk = device.pointers();
      let device = device.internal_object();
      
      check_errors(vk.MapMemory(*device, self.memory[current_buffer], buffer_offset as u64, buffer_size as u64, 0, &mut host_visible_data));
      memcpy(host_visible_data, self.data.as_ptr() as *const _, buffer_size as usize);
      let mapped_memory_range = vk::MappedMemoryRange {
        sType: vk::STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        pNext: ptr::null(),
        memory: self.memory[current_buffer],
        offset: buffer_offset as vk::DeviceSize,
        size: buffer_size as vk::DeviceSize,
      };
      vk.FlushMappedMemoryRanges(*device, 1, &mapped_memory_range);
      vk.UnmapMemory(*device, self.memory[current_buffer]);
    }
  }
  
  pub fn fill_entire_buffer_single_frame(&mut self, device: Arc<Device>, current_buffer: usize, data: Vec<T>) {
    let mut data = data;
    Buffer::align_data(Arc::clone(&device), &mut data);
    if Buffer::illegal_size(&self.size, &data) {
      return;
    }
    self.data = data;
    
    let mut host_visible_data = unsafe { mem::MaybeUninit::uninit().assume_init() };
    let buffer_size = mem::size_of::<T>() * self.data.len();
    
    unsafe {
      let vk = device.pointers();
      let device = device.internal_object();
      
      check_errors(vk.MapMemory(*device, self.memory[current_buffer], 0, buffer_size as u64, 0, &mut host_visible_data));
      memcpy(host_visible_data, self.data.as_ptr() as *const _, buffer_size as usize);
      let mapped_memory_range = vk::MappedMemoryRange {
        sType: vk::STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        pNext: ptr::null(),
        memory: self.memory[current_buffer],
        offset: 0 as vk::DeviceSize,
        size: buffer_size as vk::DeviceSize,
      };
      let mut ranges = Vec::new();
      ranges.push(mapped_memory_range);
      vk.FlushMappedMemoryRanges(*device, 1, ranges.as_ptr());
      vk.UnmapMemory(*device, self.memory[current_buffer]);
    }
  }
  
  pub fn fill_entire_buffer_all_frames(&mut self, device: Arc<Device>, data: Vec<T>) {
    let mut data = data;
    Buffer::align_data(Arc::clone(&device), &mut data);
    if Buffer::illegal_size(&self.size, &data) {
      return;
    }
    self.data = data;
    
    let mut host_visible_data = unsafe { mem::MaybeUninit::uninit().assume_init() };
    let buffer_size = mem::size_of::<T>() * self.data.len();
    
    for i in 0..self.memory.len() {
      unsafe {
        let vk = device.pointers();
        let device = device.internal_object();
        
        check_errors(vk.MapMemory(*device, self.memory[i], 0, buffer_size as u64, 0, &mut host_visible_data));
        memcpy(host_visible_data, self.data.as_ptr() as *const _, buffer_size as usize);
          let mapped_memory_range = vk::MappedMemoryRange {
          sType: vk::STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
          pNext: ptr::null(),
          memory: self.memory[i],
          offset: 0 as vk::DeviceSize,
          size: buffer_size as vk::DeviceSize,
        };
        vk.FlushMappedMemoryRanges(*device, 1, &mapped_memory_range);
        vk.UnmapMemory(*device, self.memory[i]);
      }
    }
  }
  
  pub fn copy_data_to_cpu(&mut self, device: Arc<Device>) {
    let buffer_size = self.size;//mem::size_of::<T>() * self.data.len();
    
    unsafe {
      let vk = device.pointers();
      let device = device.internal_object();
      
      check_errors(vk.MapMemory(*device, self.memory[0], 0, buffer_size as u64, 0, self.data.as_mut_ptr() as *mut *mut c_void));
    }
  }
  
  pub fn internal_object(&self, current_buffer: usize) -> &vk::Buffer {
    &self.buffer[current_buffer]
  }
  
  pub fn internal_memory(&self, current_buffer: usize) -> &vk::DeviceMemory {
    &self.memory[current_buffer]
  }
  
  pub fn internal_data(&self) -> Vec<T> {
    self.data.to_vec()
  }
  
  pub fn max_size(&self) -> u64 {
    self.size
  }
  
  fn create_buffer(instance: Arc<Instance>, device: Arc<Device>, usage: &BufferUsage, properties: vk::MemoryPropertyFlags, data_len: u64) -> (vk::Buffer, vk::DeviceMemory) {
    let mut data_len = data_len;
    Buffer::<T>::align_phantom_data(Arc::clone(&device), &mut data_len);
    let mut buffer: vk::Buffer = unsafe { mem::MaybeUninit::uninit().assume_init() };
    let mut buffer_memory: vk::DeviceMemory = unsafe { mem::MaybeUninit::uninit().assume_init() };
    
    let buffer_create_info = {
      vk::BufferCreateInfo {
        sType: vk::STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        pNext: ptr::null(),
        flags: 0,
        size: (mem::size_of::<T>() * data_len as usize) as vk::DeviceSize,
        usage: usage.to_bits(),
        sharingMode: vk::SHARING_MODE_EXCLUSIVE,
        queueFamilyIndexCount: 0,
        pQueueFamilyIndices: ptr::null(),
      }
    };
    
    let mut memory_requirements: vk::MemoryRequirements = unsafe { mem::MaybeUninit::uninit().assume_init() };
    
    unsafe {
      let vk = device.pointers();
      let device = device.internal_object();
      check_errors(vk.CreateBuffer(*device, &buffer_create_info, ptr::null(), &mut buffer));
      vk.GetBufferMemoryRequirements(*device, buffer, &mut memory_requirements);
    }
    
    let memory_type_bits_index = {
      let mut memory_properties: vk::PhysicalDeviceMemoryProperties = unsafe { mem::MaybeUninit::uninit().assume_init() };
      
      unsafe {
        let vk = instance.pointers();
        let phys_device = device.physical_device();
        vk.GetPhysicalDeviceMemoryProperties(*phys_device, &mut memory_properties);
      }
      
      let mut index: i32 = -1;
      for i in 0..memory_properties.memoryTypeCount as usize {
        if memory_requirements.memoryTypeBits & (1 << i) != 0 && memory_properties.memoryTypes[i].propertyFlags & properties == properties && (memory_properties.memoryTypes[i].propertyFlags & (0x00000080 | 0x00000040)) == 0 {
          index = i as i32;
        }
      }
      
      if index == -1 {
        panic!("Failed to find suitable memory type");
      }
      
      index
    };
    
    let memory_allocate_info = {
      vk::MemoryAllocateInfo {
        sType: vk::STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        pNext: ptr::null(),
        allocationSize: memory_requirements.size,
        memoryTypeIndex: memory_type_bits_index as u32,
      }
    };
    
    unsafe {
      let vk = device.pointers();
      let device = device.internal_object();
      check_errors(vk.AllocateMemory(*device, &memory_allocate_info, ptr::null(), &mut buffer_memory));
      vk.BindBufferMemory(*device, buffer, buffer_memory, 0);
    }
    
    (buffer, buffer_memory)
  }
  
  pub fn destroy(&self, device: Arc<Device>) {
    for i in 0..self.memory.len() {
      unsafe {
        let vk = device.pointers();
        let device = device.internal_object();
        vk.FreeMemory(*device, self.memory[i], ptr::null());
        vk.DestroyBuffer(*device, self.buffer[i], ptr::null());
      }
    }
  }
}

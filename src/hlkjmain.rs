extern crate ash;
extern crate winit;
extern crate image;

use ash::util::*;
use ash::vk;
use std::default::Default;
use std::ffi::CString;
use std::io::Cursor;
use std::mem;
use std::mem::align_of;

use winit::{
  dpi::{LogicalSize, PhysicalSize},
  event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent},
  event_loop::{ControlFlow, EventLoop},
  window::WindowBuilder
};

use std::time::Instant;

use crate::ash::version::DeviceV1_0;

//mod modules;
mod lib;

use crate::lib::modules::{VkWindow, Buffer, Shader, GraphicsPipelineBuilder, ImageBuilder, Sampler,
                     DescriptorSet, DescriptorWriter, ComputeShader};
use crate::lib::{Vulkan, find_memorytype_index};

const APP_NAME: &str = "Maat-Vulkan";
const WINDOW_SIZE: [u32; 2] = [1280, 720];

#[derive(Clone, Debug, Copy)]
pub struct Vertex {
  pos: [f32; 4],
  colour: [f32; 4],
}

#[derive(Clone, Debug, Copy)]
struct TextureVertex {
  pos: [f32; 4],
  uv: [f32; 2],
}

#[derive(Clone, Debug, Copy)]
struct UniformBuffer {
  colour: [f32; 4],
}

pub fn generate_bezier_geometry(bezier_points: Vec<[f32; 2]>) -> (Vec<u32>, Vec<Vertex>) {
  let mut indicies: Vec<u32> = Vec::new();
  let mut verticies = Vec::new();
  
  for i in 0..bezier_points.len()-1 {
    let perp_points_0 = [bezier_points[i][1], -bezier_points[i][0]];
    let perp_points_1 = [bezier_points[i+1][1], -bezier_points[i+1][0]];
    
    let mut index0 = [bezier_points[i][0] - perp_points_0[0]*0.1, bezier_points[i][1] - perp_points_0[1]*0.1];
    let mut index1 = [bezier_points[i][0] + perp_points_0[0]*0.1, bezier_points[i][1] + perp_points_0[1]*0.1];
    let mut index2 = [bezier_points[i+1][0] + perp_points_1[0]*0.1, bezier_points[i+1][1] + perp_points_1[1]*0.1];
    let mut index3 = [bezier_points[i+1][0] - perp_points_1[0]*0.1, bezier_points[i+1][1] - perp_points_1[1]*0.1];
    
    if index0 < index2 {
      
    }
    
    let mut num_indicies = indicies.len() as u32 - if indicies.len() > 0 { 3*(indicies.len() as u32/6)-1*(indicies.len() as u32/6) } else { 0 };
    indicies.push(num_indicies + 0);
    indicies.push(num_indicies + 1);
    indicies.push(num_indicies + 2);
    indicies.push(num_indicies + 2);
    indicies.push(num_indicies + 3);
    indicies.push(num_indicies + 1);
    
    verticies.push(
      Vertex {
        pos: [index0[0], index0[1], 0.0, 1.0],
        colour: [0.0, 1.0, 0.0, 1.0],
      }
    );
    
    verticies.push(
      Vertex {
        pos: [index1[0], index1[1], 0.0, 1.0],
        colour: [0.0, 0.0, 1.0, 1.0],
      }
    );
    
    verticies.push(
      Vertex {
        pos: [index2[0], index2[1], 0.0, 1.0],
        colour: [1.0, 0.0, 0.0, 1.0],
      }
    );
    
    verticies.push(
      Vertex {
        pos: [index3[0], index3[1], 0.0, 1.0],
        colour: [1.0, 0.0, 1.0, 1.0],
      }
    );
  }
  
  (indicies, verticies)
}

fn create_triangle_shader(vulkan: &Vulkan) -> (Shader<Vertex>, Buffer<u32>, Buffer<Vertex>) {
  let triangle_index_buffer_data = vec![0u32, 1, 2];
  let triangle_vertices = vec![
      Vertex {
          pos: [-1.0, 1.0, 0.0, 1.0],
          colour: [0.0, 1.0, 0.0, 1.0],
      },
      Vertex {
          pos: [1.0, 1.0, 0.0, 1.0],
          colour: [0.0, 0.0, 1.0, 1.0],
      },
      Vertex {
          pos: [0.0, -1.0, 0.0, 1.0],
          colour: [1.0, 0.0, 0.0, 1.0],
      },
  ];
  
  let triangle_vertex = Vertex {
    pos: [0.0, 0.0, 0.0, 0.0],
    colour: [0.0, 0.0, 0.0, 0.0],
  };
  
  let triangle_index_buffer = Buffer::<u32>::new_index(&vulkan.device(), triangle_index_buffer_data);
  let triangle_vertex_buffer = Buffer::<Vertex>::new_vertex(vulkan.device(), triangle_vertices);
  
  let mut graphics_pipeline_builder = GraphicsPipelineBuilder::new().topology_triangle_list()
                                                                    .front_face_counter_clockwise()
                                                                    .polygon_mode_fill()
                                                                    .samples_1();
  
  let triangle_shader = Shader::new(vulkan.device(),
                                    Cursor::new(&include_bytes!("../shaders/triangle_vert.spv")[..]),
                                    Cursor::new(&include_bytes!("../shaders/triangle_frag.spv")[..]),
                                    triangle_vertex, 
                                    vec!(offset_of!(Vertex, pos) as u32, offset_of!(Vertex, colour) as u32), 
                                    graphics_pipeline_builder,
                                    vulkan.renderpass(),
                                    vulkan.viewports(), 
                                    vulkan.scissors(),
                                    &vec![]);
  
  (triangle_shader, triangle_index_buffer, triangle_vertex_buffer)
}

fn create_texture_shader(vulkan: &Vulkan, descriptor_sets: &DescriptorSet) -> (Shader<TextureVertex>, Buffer<u32>, Buffer<TextureVertex>) {
  let texture_index_buffer_data = vec![0, 1, 2, 3, 4, 5];
  let texture_vertices = vec![
    TextureVertex {
        pos: [1.0, -1.0, 0.0, 1.0],
        uv: [0.0, 0.0],
    },
    TextureVertex {
        pos: [-1.0, -1.0, 0.0, 1.0],
        uv: [1.0, 0.0],
    },
    TextureVertex {
        pos: [-1.0, 1.0, 0.0, 1.0],
        uv: [1.0, 1.0],
    },
    
    TextureVertex {
        pos: [-1.0, 1.0, 0.0, 1.0],
        uv: [1.0, 1.0],
    },
    TextureVertex {
        pos: [1.0, 1.0, 0.0, 1.0],
        uv: [0.0, 1.0],
    },
    TextureVertex {
        pos: [1.0, -1.0, 0.0, 1.0],
        uv: [0.0, 0.0],
    }
  ];
  
  let texture_vertex = TextureVertex {
    pos: [0.0, 0.0, 0.0, 0.0],
    uv: [0.0, 0.0],
  };
  
  let texture_index_buffer = Buffer::<u32>::new_index(&vulkan.device(), texture_index_buffer_data);
  let texture_vertex_buffer = Buffer::<TextureVertex>::new_vertex(vulkan.device(), texture_vertices);
  
  let mut graphics_pipeline_builder = GraphicsPipelineBuilder::new().topology_triangle_list()
                                                                    .front_face_counter_clockwise()
                                                                    .polygon_mode_fill()
                                                                    .samples_1();
  
  let texture_shader = Shader::new(vulkan.device(),
                                    Cursor::new(&include_bytes!("../shaders/texture_vert.spv")[..]),
                                    Cursor::new(&include_bytes!("../shaders/texture_frag.spv")[..]),
                                    texture_vertex, 
                                    vec!(offset_of!(TextureVertex, pos) as u32, offset_of!(TextureVertex, uv) as u32), 
                                    graphics_pipeline_builder,
                                    vulkan.renderpass(),
                                    vulkan.viewports(), 
                                    vulkan.scissors(),
                                    descriptor_sets.layouts());
  
  (texture_shader, texture_index_buffer, texture_vertex_buffer)
}

fn main() {
  let mut screen_resolution = vk::Extent2D { width: 1, height: 1};
  
  let mut event_loop = EventLoop::new();
  let mut window = VkWindow::new(APP_NAME, WINDOW_SIZE[0], WINDOW_SIZE[1], &event_loop, &mut screen_resolution);
  
  let mut vulkan = Vulkan::new(&mut window, screen_resolution);
  
  let (triangle_shader, triangle_index_buffer, triangle_vertex_buffer) = create_triangle_shader(&vulkan);
  
  let uniform_data = vec![
    UniformBuffer {
      colour: [0.0, 0.0, 0.0, 0.0],
    }
  ];
  
  let uniform_buffer = Buffer::<UniformBuffer>::new_uniform_buffer(vulkan.device(), &uniform_data);
  
  let image = image::load_from_memory(include_bytes!("../textures/rust.png"))
                    .unwrap()
                    .fliph()
                    .to_rgba();
  
  let dimensions = image.dimensions();
  let image_data = image.into_raw();
  
  let mut src_buffer = Buffer::<u8>::new_image(vulkan.device(), image_data);
  let mut dst_image = ImageBuilder::new(vk::Format::R8G8B8A8_UNORM, 1, 1)
                                   .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                                   .set_dimensions(dimensions.0, dimensions.1)
                                   .build_device_local(vulkan.device());
  
  vulkan.copy_buffer_to_device_local_image(&src_buffer, &dst_image);
  
  let sampler = Sampler::builder()
                         .min_filter_linear()
                         .mag_filter_linear()
                         .address_mode_mirrored_repeat()
                         .mipmap_mode_linear()
                         .border_colour_float_opaque_white()
                         .compare_op_never()
                         .build(vulkan.device());
  
  let (descriptor_sets, descriptor_pool) = DescriptorSet::builder()
                                                        .uniform_buffer_fragment()
                                                        .combined_image_sampler_fragment()
                                                        .build(vulkan.device());
  let descriptor_set_writer = DescriptorWriter::builder().update_uniform_buffer(&uniform_buffer, &descriptor_sets)
                                                         .update_image(&dst_image, &sampler, &descriptor_sets);
  
  let (texture_shader, texture_index_buffer, texture_vertex_buffer) = create_texture_shader(&vulkan, &descriptor_sets);
  
  descriptor_set_writer.build(vulkan.device());
  
  let (compute_descriptor_sets, pool) = DescriptorSet::builder()
                                                      .storage_compute()
                                                      .storage_compute()
                                                      .build(vulkan.device());
  
  let compute_shader = ComputeShader::new(vulkan.device(), 
                                          Cursor::new(&include_bytes!("../shaders/bezier_comp.spv")[..]),
                                          &compute_descriptor_sets);
  
  /*
  let mut compute_data = vec![64, 32, 8, 12, 96];
  
  let x = compute_data.len() as u32;
  let y = 1;
  let z = 1;
  
  vulkan.run_compute(&compute_shader, &compute_descriptor_sets, &mut compute_data, x, y, z);
  println!("Compute Data: {:?}", compute_data);
  */
  let bezier_points = vec![ 0.0,  0.9, 
                           -0.9,  0.0, 
                            0.0, -0.9, 
                            0.9,  0.0];
  
  let mut output_data = vec![0.0; (bezier_points.len()-2)*4];
  
  let input_buffer = Buffer::<f32>::builder()
                               .data(bezier_points.to_vec())
                               .usage_transfer_src()
                               .memory_properties_host_visible_coherent()
                               .build(vulkan.device());
  
  let mut bezier_buffer = Buffer::<f32>::builder()
                                       .data(output_data.to_vec())
                                       .usage_vertex_dst_buffer()
                                       .memory_properties_host_visible_coherent()
                                       .build(vulkan.device());
  
  let x = bezier_points.len() as u32;
  let y = 1;
  let z = 1;
  
  vulkan.run_compute(&compute_shader, &compute_descriptor_sets, &input_buffer, &mut bezier_buffer, x, y, z);
  
  output_data = bezier_buffer.retrieve_buffer_data(vulkan.device());
  for i in 0..(output_data.len() as f32*0.5) as usize {
    let j = i*2;
    println!("Compute Data: x {}, y {}", output_data[j], output_data[j+1]);
  }
  //let bezier_points = vec![[0.0, 0.9], [-0.9, 0.0], [0.0, -0.9], [0.9, 0.0]];
  //let (bez_index, bez_vertex) = generate_bezier_geometry(bezier_points);
  let mut bez_index = Vec::new();
  for i in 0..(output_data.len() as f32*0.25).floor() as u32 {
    let mut j = bez_index.len() as u32 - if bez_index.len() > 0 { 3*(bez_index.len() as u32/6)-1*(bez_index.len() as u32/6) } else { 0 };
    //let j = i*7;
    bez_index.append(&mut vec![
      j, j+1, j+2, j+2, j+3, j+1
    ]);
  }
  println!("Indicies: {:?}", bez_index);
 // let bez_index = bez_index[bez_index.len()-24..].to_vec();
  
  let mut bez_vertex = Vec::new();
  for i in 0..(output_data.len() as f32*0.5) as usize {
    let j = i*2;
    
    bez_vertex.push(
      Vertex {
        pos: [output_data[j], output_data[j+1], 0.0, 1.0],
        colour: [0.0, 0.0, 0.0, 1.0],
      }
    );
  }
  
  let triangle_vertex = Vertex {
    pos: [0.0, 0.0, 0.0, 0.0],
    colour: [0.0, 0.0, 0.0, 0.0],
  };
  
  let bezier_index_buffer = Buffer::<u32>::new_index(&vulkan.device(), bez_index);
  let bezier_vertex_buffer = Buffer::<Vertex>::new_vertex(vulkan.device(), bez_vertex);
  
  event_loop.run(move |event, _, control_flow| {
      //*control_flow = ControlFlow::Wait;
      match event {
          Event::WindowEvent { event, .. } => match event {
              WindowEvent::CloseRequested => {
                  *control_flow = ControlFlow::Exit;
              },
              WindowEvent::KeyboardInput {
                  input:
                  KeyboardInput {
                      virtual_keycode: Some(VirtualKeyCode::Escape),
                      ..
                  },
                  ..
              } => {
                *control_flow = ControlFlow::Exit
              },
              WindowEvent::Resized(dimensions) => {
                println!("resized");
                vulkan.swapchain().set_screen_resolution(
                  dimensions.width,
                  dimensions.height
                );
                
                vulkan.recreate_swapchain();
                //*control_flow = ControlFlow::Wait;
              },
              _ => (),
          },
          Event::MainEventsCleared => {
            vulkan.render_triangle(
              &bezier_vertex_buffer,
              &bezier_index_buffer,
              triangle_shader.graphics_pipeline(),
            );
            /*
            vulkan.render_texture(
              &descriptor_sets,
              &texture_shader,
              &texture_vertex_buffer,
              &texture_index_buffer,
            );*/
          },
          Event::LoopDestroyed => {
            unsafe {
              vulkan.device().internal().device_wait_idle().unwrap();
              descriptor_sets.destroy(vulkan.device());
              compute_descriptor_sets.destroy(vulkan.device());
              compute_shader.destroy(vulkan.device());
              triangle_shader.destroy(vulkan.device());
              triangle_index_buffer.destroy(vulkan.device());
              triangle_vertex_buffer.destroy(vulkan.device());
              texture_shader.destroy(vulkan.device());
              texture_index_buffer.destroy(vulkan.device());
              texture_vertex_buffer.destroy(vulkan.device());
            }
          }
          _ => (),
      }
  });
}

// Copyright (c) 2016 Dominik Zeromski <dzeromsk@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <libdrm/drm.h>
#include <libdrm/intel_bufmgr.h>

#define CMD(PIPELINE, OP, SUB_OP)                                              \
  ((3 << 29) | ((PIPELINE) << 27) | ((OP) << 24) | ((SUB_OP) << 16))

#define CMD_PIPE_CONTROL CMD(3, 2, 0)
#define CMD_PIPELINE_SELECT CMD(1, 1, 4)
#define PIPELINE_SELECT_GPGPU 2
#define CMD_STATE_BASE_ADDRESS CMD(0, 1, 1)
#define CMD_MEDIA_STATE_POINTERS CMD(2, 0, 0)
#define CMD_MEDIA_CURBE_LOAD CMD(2, 0, 1)
#define CMD_MEDIA_INTERFACE_DESCRIPTOR_LOAD CMD(2, 0, 2)
#define CMD_GPGPU_WALKER CMD(2, 1, 5)
#define CMD_MEDIA_STATE_FLUSH CMD(2, 0, 4)

#define CMD_LOAD_REGISTER_IMM (0x22 << 23)
#define CMD_BATCH_BUFFER_END (0xA << 23)

// l3 cache
#define GEN8_L3_CNTL_REG_ADDRESS_OFFSET (0x7034)

#define SRFC_OFFSET (0x0400)
#define CURB_OFFSET (0x4400)
#define IDRT_OFFSET (0x8400)

typedef struct gen8_interface_descriptor {
  struct {
    uint32_t pad6 : 6;
    uint32_t kernel_start_pointer : 26;
  } desc0;
  struct {
    uint32_t kernel_start_pointer_high : 16;
    uint32_t pad6 : 16;
  } desc1;

  struct {
    uint32_t pad : 7;
    uint32_t software_exception : 1;
    uint32_t pad2 : 3;
    uint32_t maskstack_exception : 1;
    uint32_t pad3 : 1;
    uint32_t illegal_opcode_exception : 1;
    uint32_t pad4 : 2;
    uint32_t floating_point_mode : 1;
    uint32_t thread_priority : 1;
    uint32_t single_program_flow : 1;
    uint32_t denorm_mode : 1;
    uint32_t thread_preemption_disable : 1;
    uint32_t pad5 : 11;
  } desc2;

  struct {
    uint32_t pad : 2;
    uint32_t sampler_count : 3;
    uint32_t sampler_state_pointer : 27;
  } desc3;

  struct {
    uint32_t binding_table_entry_count : 5; /* prefetch entries only */
    uint32_t binding_table_pointer : 27;    /* 11 bit only on IVB+ */
  } desc4;

  struct {
    uint32_t curbe_read_offset : 16; /* in GRFs */
    uint32_t curbe_read_len : 16;    /* in GRFs */
  } desc5;

  struct {
    uint32_t group_threads_num : 10; /* 0..64, 0 - no barrier use */
    uint32_t pad : 5;
    uint32_t global_barrier_enable : 1;
    uint32_t slm_sz : 5; /* 0..16 - 0K..64K */
    uint32_t barrier_enable : 1;
    uint32_t rounding_mode : 2;
    uint32_t barrier_return_grf_offset : 8;
  } desc6;

  uint32_t desc7; /* unused */
} gen8_interface_descriptor_t;

typedef struct gen8_surface_state {
  struct {
    uint32_t cube_pos_z : 1;
    uint32_t cube_neg_z : 1;
    uint32_t cube_pos_y : 1;
    uint32_t cube_neg_y : 1;
    uint32_t cube_pos_x : 1;
    uint32_t cube_neg_x : 1;
    uint32_t media_boundary_pixel_mode : 2;
    uint32_t render_cache_rw_mode : 1;
    uint32_t sampler_L2_bypass_mode : 1;
    uint32_t vertical_line_stride_offset : 1;
    uint32_t vertical_line_stride : 1;
    uint32_t tile_mode : 2;
    uint32_t horizontal_alignment : 2;
    uint32_t vertical_alignment : 2;
    uint32_t surface_format : 9;
    uint32_t pad0 : 1;
    uint32_t surface_array : 1;
    uint32_t surface_type : 3;
  } ss0;

  struct {
    uint32_t surface_qpitch : 15;
    uint32_t pad0 : 3;
    uint32_t pad1 : 1;
    uint32_t base_mip_level : 5;
    uint32_t mem_obj_ctrl_state : 7;
    uint32_t pad2 : 1;
  } ss1;

  struct {
    uint32_t width : 14;
    uint32_t pad1 : 2;
    uint32_t height : 14;
    uint32_t pad0 : 2;
  } ss2;

  struct {
    uint32_t surface_pitch : 18;
    uint32_t pad1 : 2;
    uint32_t pad0 : 1;
    uint32_t depth : 11;
  } ss3;

  struct {
    union {
      struct {
        uint32_t multisample_pos_palette_idx : 3;
        uint32_t multisample_num : 3;
        uint32_t multisample_format : 1;
        uint32_t render_target_view_ext : 11;
        uint32_t min_array_elt : 11;
        uint32_t render_target_and_sample_rotation : 2;
        uint32_t pad1 : 1;
      };

      uint32_t pad0;
    };
  } ss4;

  struct {
    uint32_t mip_count : 4;
    uint32_t surface_min_lod : 4;
    uint32_t pad5 : 4;
    uint32_t pad4 : 2;
    uint32_t conherency_type : 1;
    uint32_t pad3 : 3;
    uint32_t pad2 : 2;
    uint32_t cube_ewa : 1;
    uint32_t y_offset : 3;
    uint32_t pad0 : 1;
    uint32_t x_offset : 7;
  } ss5;

  struct {
    union {
      union {
        struct {
          uint32_t aux_surface_mode : 3;
          uint32_t aux_surface_pitch : 9;
          uint32_t pad3 : 4;
        };
        struct {
          uint32_t uv_plane_y_offset : 14;
          uint32_t pad2 : 2;
        };
      };

      struct {
        uint32_t uv_plane_x_offset : 14;
        uint32_t pad1 : 1;
        uint32_t seperate_uv_plane_enable : 1;
      };
      struct {
        uint32_t aux_sruface_qpitch : 15;
        uint32_t pad0 : 1;
      };
    };
  } ss6;

  struct {
    uint32_t resource_min_lod : 12;
    uint32_t pad0 : 4;
    uint32_t shader_channel_select_alpha : 3;
    uint32_t shader_channel_select_blue : 3;
    uint32_t shader_channel_select_green : 3;
    uint32_t shader_channel_select_red : 3;
    uint32_t alpha_clear_color : 1;
    uint32_t blue_clear_color : 1;
    uint32_t green_clear_color : 1;
    uint32_t red_clear_color : 1;
  } ss7;

  struct {
    uint32_t surface_base_addr_lo;
  } ss8;

  struct {
    uint32_t surface_base_addr_hi;
  } ss9;

  struct {
    uint32_t pad0 : 12;
    uint32_t aux_base_addr_lo : 20;
  } ss10;

  struct {
    uint32_t aux_base_addr_hi : 32;
  } ss11;

  struct {
    uint32_t pad0;
  } ss12;

  /* 13~15 have meaning only when aux surface mode == AUX_HIZ */
  struct {
    uint32_t pad0;
  } ss13;
  struct {
    uint32_t pad0;
  } ss14;
  struct {
    uint32_t pad0;
  } ss15;
} gen8_surface_state_t;

typedef struct surface_heap {
  uint32_t binding_table[256];
  gen8_surface_state_t surface[256];
} surface_heap_t;

typedef struct gen8_sampler_state {
  struct {
    uint32_t aniso_algorithm : 1;
    uint32_t lod_bias : 13;
    uint32_t min_filter : 3;
    uint32_t mag_filter : 3;
    uint32_t mip_filter : 2;
    uint32_t base_level : 5;
    uint32_t lod_preclamp : 2;
    uint32_t default_color_mode : 1;
    uint32_t pad0 : 1;
    uint32_t disable : 1;
  } ss0;

  struct {
    uint32_t cube_control_mode : 1;
    uint32_t shadow_function : 3;
    uint32_t chromakey_mode : 1;
    uint32_t chromakey_index : 2;
    uint32_t chromakey_enable : 1;
    uint32_t max_lod : 12;
    uint32_t min_lod : 12;
  } ss1;

  struct {
    uint32_t lod_clamp_mag_mode : 1;
    uint32_t flexible_filter_valign : 1;
    uint32_t flexible_filter_halign : 1;
    uint32_t flexible_filter_coeff_size : 1;
    uint32_t flexible_filter_mode : 1;
    uint32_t pad1 : 1;
    uint32_t indirect_state_ptr : 18;
    uint32_t pad0 : 2;
    uint32_t sep_filter_height : 2;
    uint32_t sep_filter_width : 2;
    uint32_t sep_filter_coeff_table_size : 2;
  } ss2;

  struct {
    uint32_t r_wrap_mode : 3;
    uint32_t t_wrap_mode : 3;
    uint32_t s_wrap_mode : 3;
    uint32_t pad : 1;
    uint32_t non_normalized_coord : 1;
    uint32_t trilinear_quality : 2;
    uint32_t address_round : 6;
    uint32_t max_aniso : 3;
    uint32_t pad0 : 2;
    uint32_t non_sep_filter_footprint_mask : 8;
  } ss3;
} gen8_sampler_state_t;

typedef struct gen7_sampler_border_color {
  float r, g, b, a;
} gen7_sampler_border_color_t;

static void setup_input(uint8_t *data) {
  int *input = (int *)data;
  int i;
  for (i = 0; i < 64; i++)
    input[i] = i;
}

char kernel[] = {
    // mov (16) r1.0<1>:uw 0xffff:uw { align1, h1, nomask }
    "\x01\x00\x80\x00\x4c\x16\x20\x20\x00\x00\x00\x10\xff\xff\x00\x00"

    // mov (16) r1.0<1>:uw 0x0000:uw { align1, h1 }
    "\x01\x00\x80\x00\x48\x16\x20\x20\x00\x00\x00\x10\x00\x00\x00\x00"

    // mov (1) r8.0<2>:uw 0x0000:uw { align1, q1 }
    "\x01\x00\x00\x00\x48\x16\x00\x41\x00\x00\x00\x10\x00\x00\x00\x00"

    // mov (1) r8.2<2>:uw 0xffff:w { align1, q1 }
    "\x01\x00\x00\x00\x48\x1e\x04\x41\x00\x00\x00\x18\xff\xff\xff\xff"

    // cmp.le.f0.0 (16) null:uw r1.0<8;8,1>:uw 0x0000:uw { align1, h1, switch,
    // nomask }
    "\x10\x80\x80\x06\x44\x12\x00\x20\x20\x00\x8d\x16\x00\x00\x00\x00"

    // (+f0.0) if (16) 208 208 0 { align1, h1 }
    "\x22\x00\x81\x00\x00\x06\x00\x20\xd0\x00\x00\x00\xd0\x00\x00\x00"

    // mul (1) r127.7<1>:d r0.1<0;1,0>:d r8.6<0;1,0>:ud { align1, q1, nomask }
    "\x41\x00\x00\x00\x2c\x0a\xfc\x2f\x04\x00\x00\x02\x18\x01\x00\x00"

    // add (1) r127.6<1>:d r8.7<0;1,0>:d r127.7<0;1,0>:d { align1, q1, nomask }
    "\x40\x00\x00\x00\x2c\x0a\xf8\x2f\x1c\x01\x00\x0a\xfc\x0f\x00\x00"

    // add (16) r124.0<1>:d r127.6<0;1,0>:d r2.0<8;8,1>:d { align1, h1 }
    "\x40\x00\x80\x00\x28\x0a\x80\x2f\xf8\x0f\x00\x0a\x40\x00\x8d\x00"

    // mul (16) r122.0<1>:d r124.0<8;8,1>:d 0x0004:w { align1, h1 }
    "\x41\x00\x80\x00\x28\x0a\x40\x2f\x80\x0f\x8d\x1e\x04\x00\x00\x00"

    // mul (16) r110.0<1>:d r124.0<8;8,1>:d 0x0004:w { align1, h1 }
    "\x41\x00\x80\x00\x28\x0a\xc0\x2d\x80\x0f\x8d\x1e\x04\x00\x00\x00"

    // add (16) r120.0<1>:d r8.2<0;1,0>:d r122.0<8;8,1>:d { align1, h1 }
    "\x40\x00\x80\x00\x28\x0a\x00\x2f\x08\x01\x00\x0a\x40\x0f\x8d\x00"

    // add (16) r108.0<1>:d r8.4<0;1,0>:d r110.0<8;8,1>:d { align1, h1 }
    "\x40\x00\x80\x00\x28\x0a\x80\x2d\x10\x01\x00\x0a\xc0\x0d\x8d\x00"

    // add (16) r118.0<1>:ud r120.0<8;8,1>:ud -r8.2<0;1,0>:ud { align1, h1,
    // nomask }
    "\x40\x00\x80\x00\x0c\x02\xc0\x2e\x00\x0f\x8d\x02\x08\x41\x00\x00"

    // add (16) r112.0<1>:ud r108.0<8;8,1>:ud -r8.4<0;1,0>:ud { align1, h1,
    // nomask }
    "\x40\x00\x80\x00\x0c\x02\x00\x2e\x80\x0d\x8d\x02\x10\x41\x00\x00"

    // send (16) r116.0<1>:uw r118 0x0c 0x04205e02:d  [ data cache data port 1,
    // msg-length:2, resp-length:2, header:no, untyped surface read,
    // mode:simd16, channels:r, bti:2 ] { align1, h1 }
    "\x31\x00\x80\x0c\x48\x02\x80\x2e\xc0\x0e\x8d\x0e\x02\x5e\x20\x04"

    // shl (16) r114.0<1>:d r116.0<8;8,1>:d 0x00000001:d { align1, h1 }
    "\x09\x00\x80\x00\x28\x0a\x40\x2e\x80\x0e\x8d\x0e\x01\x00\x00\x00"

    // send (16) null:uw r112 0x0c 0x08025e03:d  [ data cache data port 1,
    // msg-length:4, resp-length:0, header:no, untyped surface write,
    // mode:simd16, channels:r, bti:3 ] { align1, h1 }
    "\x31\x00\x80\x0c\x40\x02\x00\x20\x00\x0e\x8d\x0e\x03\x5e\x02\x08"

    // endif (16) 0 { align1, h1 }
    "\x25\x00\x80\x00\x00\x00\x00\x20\x00\x00\x8d\x0e\x00\x00\x00\x00"

    // mov (16) r112.0<1>:ud r0.0<8;8,1>:ud { align1, h1, nomask }
    "\x01\x00\x80\x00\x0c\x02\x00\x2e\x00\x00\x8d\x00\x00\x00\x00\x00"

    // send (8) null:ud r112 0x27 0x02000010:ud  [ thread spawner, msg-length:1,
    // resp-length:0, header:no, func-control:0x00010 ] { align1, q1, eot }
    "\x31\x00\x60\x07\x00\x02\x00\x20\x00\x0e\x8d\x06\x10\x00\x00\x82"

};

static void setup_kernel0(uint8_t *data) {
  memcpy(data, kernel, sizeof kernel);
}

static void setup_curb0(uint8_t *data) {
  int *curb = (int *)data;
  int i, j;
  int id_offset = 8;
  int count_offset = 60;
  for (i = 0; i < 4; i++) {
    int slice = i * 64;
    for (j = 0; j < 16; j++) {
      curb[slice + id_offset + j] = j + (i * 16);
    }
    // curb[slice + count_offset] = 64;
  }
}

static void setup_idrt0(uint8_t *data) {
  gen8_interface_descriptor_t *idrt = (gen8_interface_descriptor_t *)data;
  // idrt[0].desc3.sampler_state_pointer = 1088;
  idrt[0].desc5.curbe_read_len = 8;
  idrt[0].desc6.group_threads_num = 4;
}

static void setup_heap0(uint8_t *data) {
  uint32_t *bind = (uint32_t *)data;
  gen8_surface_state_t *srfc = (gen8_surface_state_t *)(data + 1024);

  bind[2] = 1152;
  bind[3] = 1216;

  srfc[2].ss0.surface_format = 511;
  srfc[2].ss0.surface_type = 4;
  srfc[2].ss1.mem_obj_ctrl_state = 120;
  srfc[2].ss2.width = 127;
  srfc[2].ss2.height = 1;
  srfc[3].ss0.surface_format = 511;
  srfc[3].ss0.surface_type = 4;
  srfc[3].ss1.mem_obj_ctrl_state = 120;
  srfc[3].ss2.width = 127;
  srfc[3].ss2.height = 1;
}

static void setup_batch0(uint8_t *data) {
  uint32_t *batch = (uint32_t *)data;
  int i = 0;

#define OUT_BATCH(x) batch[i++] = x

  OUT_BATCH(CMD_PIPE_CONTROL | 4);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00101420);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);

  OUT_BATCH(CMD_LOAD_REGISTER_IMM | 1);
  OUT_BATCH(GEN8_L3_CNTL_REG_ADDRESS_OFFSET);
  OUT_BATCH(0x60000160);

  OUT_BATCH(CMD_PIPE_CONTROL | 4);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00101420);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);

  OUT_BATCH(CMD_PIPELINE_SELECT | PIPELINE_SELECT_GPGPU);

  OUT_BATCH(CMD_STATE_BASE_ADDRESS | 14);
  OUT_BATCH(0x00000781);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00780000);
  OUT_BATCH(0x00000781);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000781);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000781);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000781);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0xfffff001);
  OUT_BATCH(0xfffff001);
  OUT_BATCH(0xfffff001);
  OUT_BATCH(0xfffff001);

  OUT_BATCH(CMD_MEDIA_STATE_POINTERS | (9 - 2));
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x014f02c0);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00020200);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);

  OUT_BATCH(CMD_MEDIA_CURBE_LOAD | (4 - 2));
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00004000);
  OUT_BATCH(0x00004400);

  OUT_BATCH(CMD_MEDIA_INTERFACE_DESCRIPTOR_LOAD | (4 - 2));
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000020);
  OUT_BATCH(0x00008400);

  OUT_BATCH(CMD_GPGPU_WALKER | 13);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x40000003);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000001);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000001);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000001);
  OUT_BATCH(0x0000ffff);
  OUT_BATCH(0xffffffff);

  OUT_BATCH(CMD_MEDIA_STATE_FLUSH | 0);
  OUT_BATCH(0);

  OUT_BATCH(CMD_PIPE_CONTROL | 4);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00101420);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);

  OUT_BATCH(CMD_BATCH_BUFFER_END);
}

int main(int argc, char *argv[]) {
  uint8_t kernel_data[4096] = {0};
  uint8_t state_data[40960] = {0};
  uint8_t batch_data[4096] = {0};
  uint8_t input_data[256] = {0};

  int i;
  int err;

  int fd = open("/dev/dri/card0", O_RDWR);
  drm_intel_bufmgr *bufmgr = drm_intel_bufmgr_gem_init(fd, 16384);
  drm_intel_context *ctx = drm_intel_gem_context_create(bufmgr);

  drm_intel_bo *kernel_buffer =
      drm_intel_bo_alloc(bufmgr, "kernel buffer", 464, 64);
  setup_kernel0(kernel_data);
  err = drm_intel_bo_subdata(kernel_buffer, 0, 464, kernel_data);

  drm_intel_bo *input_buffer =
      drm_intel_bo_alloc(bufmgr, "input buffer", 256, 64);

  setup_input(input_data);
  err = drm_intel_bo_subdata(input_buffer, 0, 256, input_data);

  drm_intel_bo *output_buffer =
      drm_intel_bo_alloc(bufmgr, "output buffer", 256, 64);

  drm_intel_bo *state_buffer =
      drm_intel_bo_alloc(bufmgr, "state buffer", 36864, 4096);

  setup_heap0(state_data);
  setup_curb0(state_data + CURB_OFFSET);
  setup_idrt0(state_data + IDRT_OFFSET);
  err = drm_intel_bo_subdata(state_buffer, 0, 36864, state_data);

  // surface relocations
  err = drm_intel_bo_emit_reloc(state_buffer, SRFC_OFFSET + 160, input_buffer,
                                0, 2, 2);
  err = drm_intel_bo_emit_reloc(state_buffer, SRFC_OFFSET + 160 + 64,
                                output_buffer, 0, 2, 2);

  // curb relocations
  for (i = 0; i < 4; i++) {
    int input_offset = CURB_OFFSET + i * 256 + sizeof(uint32_t) * 58;
    int output_offset = CURB_OFFSET + i * 256 + sizeof(uint32_t) * 58 + 8;
    err = drm_intel_bo_emit_reloc(state_buffer, input_offset, input_buffer, 0,
                                  2, 2);
    err = drm_intel_bo_emit_reloc(state_buffer, output_offset, output_buffer, 0,
                                  2, 2);
  }

  drm_intel_bo *batch_buffer =
      drm_intel_bo_alloc(bufmgr, "batch buffer", 512, 64);
  setup_batch0(batch_data);
  err = drm_intel_bo_subdata(batch_buffer, 0, 512, batch_data);

  // batch buffer relocations
  err = drm_intel_bo_emit_reloc(batch_buffer, 20 * sizeof(uint32_t),
                                state_buffer, 1921, 4, 4);
  err = drm_intel_bo_emit_reloc(batch_buffer, 22 * sizeof(uint32_t),
                                state_buffer, 1921, 2, 2);
  err = drm_intel_bo_emit_reloc(batch_buffer, 26 * sizeof(uint32_t),
                                kernel_buffer, 1921, 16, 16);

  err = drm_intel_gem_bo_context_exec(batch_buffer, ctx, 296, 1);
  err = drm_intel_bo_busy(batch_buffer);
  drm_intel_bo_wait_rendering(batch_buffer);
  drm_intel_gem_bo_start_gtt_access(batch_buffer, 1);

  void *output_data = malloc(256);
  err = drm_intel_bo_get_subdata(output_buffer, 0, 256, output_data);

  drm_intel_bo_unreference(kernel_buffer);
  drm_intel_bo_unreference(input_buffer);
  drm_intel_bo_unreference(output_buffer);
  drm_intel_bo_unreference(state_buffer);
  drm_intel_bo_unreference(batch_buffer);
  drm_intel_gem_context_destroy(ctx);
  drm_intel_bufmgr_destroy(bufmgr);

  int *input = (int *)input_data;
  int *output = output_data;
  int correct = 0;
  for (i = 0; i < 64; i++) {
    if (output[i] == input[i] + input[i])
      correct++;
  }
  fprintf(stderr, "Computed '%d/%d' correct values!\n", correct, 64);

  return 0;
}

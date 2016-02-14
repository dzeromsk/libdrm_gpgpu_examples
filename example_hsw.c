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

// HSW+
#define HSW_SCRATCH1_OFFSET (0xB038)
#define HSW_ROW_CHICKEN3_HDC_OFFSET (0xE49C)

// L3 cache
#define GEN7_L3_SQC_REG1_ADDRESS_OFFSET (0XB010)
#define GEN7_L3_CNTL_REG2_ADDRESS_OFFSET (0xB020)
#define GEN7_L3_CNTL_REG3_ADDRESS_OFFSET (0xB024)

#define SRFC_OFFSET (0x0400)
#define CURB_OFFSET (0x4400)
#define IDRT_OFFSET (0x8400)

typedef struct gen6_interface_descriptor {
  struct {
    uint32_t pad6 : 6;
    uint32_t kernel_start_pointer : 26;
  } desc0;

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
    uint32_t pad5 : 1;
    uint32_t pad6 : 6;
    uint32_t pad7 : 6;
  } desc1;

  struct {
    uint32_t pad : 2;
    uint32_t sampler_count : 3;
    uint32_t sampler_state_pointer : 27;
  } desc2;

  struct {
    uint32_t binding_table_entry_count : 5; /* prefetch entries only */
    uint32_t binding_table_pointer : 27;    /* 11 bit only on IVB+ */
  } desc3;

  struct {
    uint32_t curbe_read_offset : 16; /* in GRFs */
    uint32_t curbe_read_len : 16;    /* in GRFs */
  } desc4;

  struct {
    uint32_t group_threads_num : 8; /* 0..64, 0 - no barrier use */
    uint32_t barrier_return_byte : 8;
    uint32_t slm_sz : 5; /* 0..16 - 0K..64K */
    uint32_t barrier_enable : 1;
    uint32_t rounding_mode : 2;
    uint32_t barrier_return_grf_offset : 8;
  } desc5;

  uint32_t desc6; /* unused */
  uint32_t desc7; /* unused */
} gen6_interface_descriptor_t;

typedef struct gen7_surface_state {
  struct {
    uint32_t cube_pos_z : 1;
    uint32_t cube_neg_z : 1;
    uint32_t cube_pos_y : 1;
    uint32_t cube_neg_y : 1;
    uint32_t cube_pos_x : 1;
    uint32_t cube_neg_x : 1;
    uint32_t media_boundary_pixel_mode : 2;
    uint32_t render_cache_rw_mode : 1;
    uint32_t pad1 : 1;
    uint32_t surface_array_spacing : 1;
    uint32_t vertical_line_stride_offset : 1;
    uint32_t vertical_line_stride : 1;
    uint32_t tile_walk : 1;
    uint32_t tiled_surface : 1;
    uint32_t horizontal_alignment : 1;
    uint32_t vertical_alignment : 2;
    uint32_t surface_format : 9;
    uint32_t pad0 : 1;
    uint32_t surface_array : 1;
    uint32_t surface_type : 3;
  } ss0;

  struct {
    uint32_t base_addr;
  } ss1;

  struct {
    uint32_t width : 14;
    uint32_t pad1 : 2;
    uint32_t height : 14;
    uint32_t pad0 : 2;
  } ss2;

  struct {
    uint32_t pitch : 18;
    uint32_t pad0 : 3;
    uint32_t depth : 11;
  } ss3;

  union {
    struct {
      uint32_t mulsample_pal_idx : 3;
      uint32_t numer_mulsample : 3;
      uint32_t mss_fmt : 1;
      uint32_t rt_view_extent : 11;
      uint32_t min_array_element : 11;
      uint32_t rt_rotate : 2;
      uint32_t pad0 : 1;
    } not_str_buf;
  } ss4;

  struct {
    uint32_t mip_count : 4;
    uint32_t surface_min_load : 4;
    uint32_t pad2 : 6;
    uint32_t coherence_type : 1;
    uint32_t stateless_force_write_thru : 1;
    uint32_t cache_control : 4;
    uint32_t y_offset : 4;
    uint32_t pad0 : 1;
    uint32_t x_offset : 7;
  } ss5;

  uint32_t ss6; /* unused */

  struct {
    uint32_t min_lod : 12;
    uint32_t pad0 : 4;
    uint32_t shader_a : 3;
    uint32_t shader_b : 3;
    uint32_t shader_g : 3;
    uint32_t shader_r : 3;
    uint32_t pad1 : 4;
  } ss7;
} gen7_surface_state_t;

typedef struct surface_heap {
  uint32_t binding_table[256];
  // char surface[16384]; // 256*sizeof(gen7_surface_state_t)
  gen7_surface_state_t surface[256];
} surface_heap_t;

typedef struct gen7_sampler_state {
  struct {
    uint32_t aniso_algorithm : 1;
    uint32_t lod_bias : 13;
    uint32_t min_filter : 3;
    uint32_t mag_filter : 3;
    uint32_t mip_filter : 2;
    uint32_t base_level : 5;
    uint32_t pad1 : 1;
    uint32_t lod_preclamp : 1;
    uint32_t default_color_mode : 1;
    uint32_t pad0 : 1;
    uint32_t disable : 1;
  } ss0;

  struct {
    uint32_t cube_control_mode : 1;
    uint32_t shadow_function : 3;
    uint32_t pad : 4;
    uint32_t max_lod : 12;
    uint32_t min_lod : 12;
  } ss1;

  struct {
    uint32_t pad : 5;
    uint32_t default_color_pointer : 27;
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
    uint32_t chroma_key_mode : 1;
    uint32_t chroma_key_index : 2;
    uint32_t chroma_key_enable : 1;
    uint32_t pad0 : 6;
  } ss3;
} gen7_sampler_state_t;

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
    "\x01\x02\x80\x00\x69\x21\x20\x20\x00\x00\x00\x00\xff\xff\x00\x00"

    // mov (16) r1.0<1>:uw 0x0000:uw { align1, h1 }
    "\x01\x00\x80\x00\x69\x21\x20\x20\x00\x00\x00\x00\x00\x00\x00\x00"

    // mov (1) r8.0<2>:uw 0x0000:uw { align1, q1 }
    "\x01\x00\x00\x00\x69\x21\x00\x41\x00\x00\x00\x00\x00\x00\x00\x00"

    // mov (1) r8.2<2>:uw 0xffff:w { align1, q1 }
    "\x01\x00\x00\x00\xe9\x31\x04\x41\x00\x00\x00\x00\xff\xff\xff\xff"

    // cmp.le.f0.0 (16) null:uw r1.0<8;8,1>:uw 0x0000:uw { align1, h1, switch,
    // nomask }
    "\x10\x82\x80\x06\x28\x2d\x00\x20\x20\x00\x8d\x00\x00\x00\x00\x00"

    // (+f0.0) if (16) 21 21 { align1, h1 }
    "\x22\x00\x81\x00\x00\x1c\x00\x20\x00\x00\x8d\x00\x15\x00\x15\x00"

    // mul (1) r127.7<1>:d r0.1<0;1,0>:d r8.4<0;1,0>:ud { align1, q1, nomask }
    "\x41\x02\x00\x00\xa5\x04\xfc\x2f\x04\x00\x00\x00\x10\x01\x00\x00"

    // add (1) r127.6<1>:d r8.5<0;1,0>:d r127.7<0;1,0>:d { align1, q1, nomask }
    "\x40\x02\x00\x00\xa5\x14\xf8\x2f\x14\x01\x00\x00\xfc\x0f\x00\x00"

    // add (16) r124.0<1>:d r127.6<0;1,0>:d r2.0<8;8,1>:d { align1, h1 }
    "\x40\x00\x80\x00\xa5\x14\x80\x2f\xf8\x0f\x00\x00\x40\x00\x8d\x00"

    // mul (16) r122.0<1>:d r124.0<8;8,1>:d 0x0004:w { align1, h1 }
    "\x41\x00\x80\x00\xa5\x3c\x40\x2f\x80\x0f\x8d\x00\x04\x00\x00\x00"

    // mul (16) r110.0<1>:d r124.0<8;8,1>:d 0x0004:w { align1, h1 }
    "\x41\x00\x80\x00\xa5\x3c\xc0\x2d\x80\x0f\x8d\x00\x04\x00\x00\x00"

    // add (16) r120.0<1>:d r8.2<0;1,0>:d r122.0<8;8,1>:d { align1, h1,
    // compacted }
    "\x40\x96\x19\x20\xe0\x78\x08\x7a"

    // add (16) r108.0<1>:d r8.3<0;1,0>:d r110.0<8;8,1>:d { align1, h1,
    // compacted }
    "\x40\x96\x1d\x20\xe0\x6c\x08\x6e"

    // add (16) r118.0<1>:ud r120.0<8;8,1>:ud -r8.2<0;1,0>:ud { align1, h1,
    // nomask, compacted }
    "\x40\x37\x5d\x20\x0f\x76\x78\x08"

    // add (16) r112.0<1>:ud r108.0<8;8,1>:ud -r8.3<0;1,0>:ud { align1, h1,
    // nomask, compacted }
    "\x40\x37\x65\x20\x0f\x70\x6c\x08"

    // send (16) r116.0<1>:uw r118 0x0c 0x04205e02:d  [ data cache data port 1,
    // msg-length:2, resp-length:2, header:no, untyped surface read,
    // mode:simd16, channels:r, bti:2 ] { align1, h1 }
    "\x31\x00\x80\x0c\x29\x1c\x80\x2e\xc0\x0e\x8d\x00\x02\x5e\x20\x04"

    // shl (16) r114.0<1>:d r116.0<8;8,1>:d 0x00000001:d { align1, h1, compacted
    // }
    "\x09\xd6\x01\x20\x07\x72\x74\x01"

    // send (16) null:uw r112 0x0c 0x08025e03:d  [ data cache data port 1,
    // msg-length:4, resp-length:0, header:no, untyped surface write,
    // mode:simd16, channels:r, bti:3 ] { align1, h1 }
    "\x31\x00\x80\x0c\x28\x1c\x00\x20\x00\x0e\x8d\x00\x03\x5e\x02\x08"

    // endif (16) 0 { align1, h1 }
    "\x25\x00\x80\x00\x00\x1c\x00\x20\x00\x00\x8d\x00\x00\x00\x00\x00"

    // mov (16) r112.0<1>:ud r0.0<8;8,1>:ud { align1, h1, nomask, compacted }
    "\x01\x57\x00\x20\x07\x70\x00\x00"

    // send (8) null:ud r112 0x27 0x02000010:ud  [ thread spawner, msg-length:1,
    // resp-length:0, header:no, func-control:0x00010 ] { align1, q1, eot }
    "\x31\x00\x60\x07\x20\x0c\x00\x20\x00\x0e\x8d\x00\x10\x00\x00\x82"

};

static void setup_kernel(uint8_t *data) { memcpy(data, kernel, sizeof kernel); }

static void setup_curb(uint8_t *data) {
  int *curb = (int *)data;
  int i, j;
  int id_offset = 8;
  int count_offset = 60;
  for (i = 0; i < 4; i++) {
    int slice = i * 64;
    for (j = 0; j < 16; j++) {
      curb[slice + id_offset + j] = j + (i * 16);
    }
    curb[slice + count_offset] = 64;
  }
}

static void setup_idrt(uint8_t *data) {
  gen6_interface_descriptor_t *idrt = (gen6_interface_descriptor_t *)data;
  // idrt[0].desc2.sampler_state_pointer = 1088;
  idrt[0].desc4.curbe_read_len = 8;
  idrt[0].desc5.slm_sz = 1;
}

static void setup_heap(uint8_t *data) {
  uint32_t *bind = (uint32_t *)data;
  gen7_surface_state_t *srfc = (gen7_surface_state_t *)(data + 1024);

  bind[2] = 1088;
  bind[3] = 1120;

  srfc[2].ss0.surface_format = 511;
  srfc[2].ss0.surface_type = 4;
  srfc[2].ss2.width = 127;
  srfc[2].ss2.height = 1;
  srfc[2].ss5.cache_control = 5;
  srfc[3].ss0.surface_format = 511;
  srfc[3].ss0.surface_type = 4;
  srfc[3].ss2.width = 127;
  srfc[3].ss2.height = 1;
  srfc[3].ss5.cache_control = 5;
}

static void setup_batch(uint8_t *data) {
  uint32_t *batch = (uint32_t *)data;
  int i = 0;

#define OUT_BATCH(x) batch[i++] = x

  OUT_BATCH(CMD_PIPE_CONTROL | 3);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00100020);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);

  OUT_BATCH(CMD_PIPE_CONTROL | 3);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00101400);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);

  OUT_BATCH(CMD_LOAD_REGISTER_IMM | 1);
  OUT_BATCH(HSW_SCRATCH1_OFFSET);
  OUT_BATCH(0x00000000);

  OUT_BATCH(CMD_LOAD_REGISTER_IMM | 1);
  OUT_BATCH(HSW_ROW_CHICKEN3_HDC_OFFSET);
  OUT_BATCH((1 << 6ul) << 16);

  OUT_BATCH(CMD_LOAD_REGISTER_IMM | 1);
  OUT_BATCH(GEN7_L3_SQC_REG1_ADDRESS_OFFSET);
  OUT_BATCH(0x08800000);

  OUT_BATCH(CMD_LOAD_REGISTER_IMM | 1);
  OUT_BATCH(GEN7_L3_CNTL_REG2_ADDRESS_OFFSET);
  OUT_BATCH(0x02000030);

  OUT_BATCH(CMD_LOAD_REGISTER_IMM | 1);
  OUT_BATCH(GEN7_L3_CNTL_REG3_ADDRESS_OFFSET);
  OUT_BATCH(0x00040410);

  OUT_BATCH(CMD_PIPE_CONTROL | 3);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00100020);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);

  OUT_BATCH(CMD_PIPE_CONTROL | 3);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00101400);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);

  OUT_BATCH(CMD_PIPELINE_SELECT | PIPELINE_SELECT_GPGPU);

  OUT_BATCH(CMD_STATE_BASE_ADDRESS | 8);
  OUT_BATCH(0x00000551);
  OUT_BATCH(0x00000551);
  OUT_BATCH(0x00000501);
  OUT_BATCH(0x00000501);
  OUT_BATCH(0x00000501);
  OUT_BATCH(0x00000001);
  OUT_BATCH(0xfffff001);
  OUT_BATCH(0x00000001);
  OUT_BATCH(0x00000001);

  OUT_BATCH(CMD_MEDIA_STATE_POINTERS | 6);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x008b00c4);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000200);
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

  OUT_BATCH(CMD_GPGPU_WALKER | 9);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x40000003);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000001);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000001);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000001);
  OUT_BATCH(0x0000ffff);
  OUT_BATCH(0xffffffff);

  OUT_BATCH(CMD_MEDIA_STATE_FLUSH | 0);
  OUT_BATCH(0);

  OUT_BATCH(CMD_PIPE_CONTROL | 3);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00100020);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);

  OUT_BATCH(CMD_PIPE_CONTROL | 3);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00101400);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);

  OUT_BATCH(CMD_LOAD_REGISTER_IMM | 1);
  OUT_BATCH(HSW_SCRATCH1_OFFSET);
  OUT_BATCH(0x00000000);

  OUT_BATCH(CMD_LOAD_REGISTER_IMM | 1);
  OUT_BATCH(HSW_ROW_CHICKEN3_HDC_OFFSET);
  OUT_BATCH((1 << 6ul) << 16);

  OUT_BATCH(CMD_LOAD_REGISTER_IMM | 1);
  OUT_BATCH(GEN7_L3_SQC_REG1_ADDRESS_OFFSET);
  OUT_BATCH(0x08800000);

  OUT_BATCH(CMD_LOAD_REGISTER_IMM | 1);
  OUT_BATCH(GEN7_L3_CNTL_REG2_ADDRESS_OFFSET);
  OUT_BATCH(0x02000030);

  OUT_BATCH(CMD_LOAD_REGISTER_IMM | 1);
  OUT_BATCH(GEN7_L3_CNTL_REG3_ADDRESS_OFFSET);
  OUT_BATCH(0x00040410);

  OUT_BATCH(CMD_PIPE_CONTROL | 3);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00100020);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00000000);

  OUT_BATCH(CMD_PIPE_CONTROL | 3);
  OUT_BATCH(0x00000000);
  OUT_BATCH(0x00101400);
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
      drm_intel_bo_alloc(bufmgr, "kernel buffer", 416, 64);
  setup_kernel(kernel_data);
  err = drm_intel_bo_subdata(kernel_buffer, 0, 416, kernel_data);

  drm_intel_bo *input_buffer =
      drm_intel_bo_alloc(bufmgr, "input buffer", 256, 64);
  setup_input(input_data);
  err = drm_intel_bo_subdata(input_buffer, 0, 256, input_data);

  drm_intel_bo *output_buffer =
      drm_intel_bo_alloc(bufmgr, "output buffer", 256, 64);

  drm_intel_bo *state_buffer =
      drm_intel_bo_alloc(bufmgr, "state buffer", 36864, 4096);
  setup_heap(state_data);
  setup_curb(state_data + CURB_OFFSET);
  setup_idrt(state_data + IDRT_OFFSET);
  err = drm_intel_bo_subdata(state_buffer, 0, 36864, state_data);

  // surface relocations
  err = drm_intel_bo_emit_reloc(state_buffer, SRFC_OFFSET + 68, input_buffer, 0,
                                2, 2);
  err = drm_intel_bo_emit_reloc(state_buffer, SRFC_OFFSET + 68 + 32,
                                output_buffer, 0, 2, 2);

  // idrt relocations
  err = drm_intel_bo_emit_reloc(state_buffer, IDRT_OFFSET, kernel_buffer, 0, 16,
                                0);
  // err = drm_intel_bo_emit_reloc(state_buffer, 33800, state_buffer, 34816, 4,
  // 0);

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
  setup_batch(batch_data);
  err = drm_intel_bo_subdata(batch_buffer, 0, 512, batch_data);

  // batch relocations
  err = drm_intel_bo_emit_reloc(batch_buffer, 38 * sizeof(uint32_t),
                                state_buffer, 1361, 16, 16);
  err = drm_intel_bo_emit_reloc(batch_buffer, 57 * sizeof(uint32_t),
                                state_buffer, CURB_OFFSET, 16, 0);
  err = drm_intel_bo_emit_reloc(batch_buffer, 61 * sizeof(uint32_t),
                                state_buffer, IDRT_OFFSET, 16, 0);

  err = drm_intel_gem_bo_context_exec(batch_buffer, ctx, 448, 1);
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

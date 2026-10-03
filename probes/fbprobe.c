/*
 * fbprobe.c — raw /dev/fb0 probe (README "Task 3"; plan slices S4/S5 verification).
 *
 * Plain Linux only: std C + POSIX. No libakuma, no Akuma-private syscall
 * (README rule 1). The fbdev structs and ioctl numbers are declared by hand
 * — exactly like src/fb.rs does in Rust, byte for byte against
 * <linux/fb.h> — so one binary runs unchanged on the kernel under test and
 * on a real Linux control box, and a byte diff of the printed structs means
 * something (the fbstress/md5probe same-binary calibration, plan §4).
 *
 * Build (on the box):  x86_64-linux-musl-gcc -O1 -static -o fbprobe fbprobe.c
 * Build (any Linux):   gcc -O1 -static -Wall -Wextra -o fbprobe fbprobe.c
 *
 * What it does:
 *   1. open("/dev/fb0", O_RDWR)
 *   2. FBIOGET_VSCREENINFO + FBIOGET_FSCREENINFO, printing every field
 *   3. mmap(smem_len, MAP_SHARED, PROT_READ|PROT_WRITE)
 *   4. fill the whole mapping with a self-identifying gradient
 *      (word i = 0xFF000000 | i), timed with clock_gettime(CLOCK_MONOTONIC)
 *   5. read every word back and compare, printing a few sampled pixels
 *   6. exit 0 only if everything above worked
 *
 * Reading the MB/s line (plan §4): ≈3000 MB/s means the user mapping is
 * write-combining (kernel S2 did its job). ≈71 MB/s means the mapping came
 * up uncached — the PAT bit is missing from the user PTEs, an S2 failure,
 * not a probe bug.
 */

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

/* The ioctl numbers, ours not the header's (kept identical to src/fb.rs). */
#define FBIOGET_VSCREENINFO 0x4600
#define FBIOGET_FSCREENINFO 0x4602

struct fb_bitfield {
	uint32_t offset;
	uint32_t length;
	uint32_t msb_right;
};

/* Mirrors struct fb_var_screeninfo: every field is u32, 160 bytes. */
struct fb_var_screeninfo {
	uint32_t xres;
	uint32_t yres;
	uint32_t xres_virtual;
	uint32_t yres_virtual;
	uint32_t xoffset;
	uint32_t yoffset;
	uint32_t bits_per_pixel;
	uint32_t grayscale;
	struct fb_bitfield red;
	struct fb_bitfield green;
	struct fb_bitfield blue;
	struct fb_bitfield transp;
	uint32_t nonstd;
	uint32_t activate;
	uint32_t height;
	uint32_t width;
	uint32_t accel_flags;
	uint32_t pixclock;
	uint32_t left_margin;
	uint32_t right_margin;
	uint32_t upper_margin;
	uint32_t lower_margin;
	uint32_t hsync_len;
	uint32_t vsync_len;
	uint32_t sync;
	uint32_t vmode;
	uint32_t rotate;
	uint32_t colorspace;
	uint32_t reserved[4];
};

/* Mirrors struct fb_fix_screeninfo (LP64 / x86_64 shape, 80 bytes).
 * Note the three __u16 pan steps — src/fb.rs once had these as u32, which
 * moved line_length to offset 52 and read the alignment padding instead of
 * the pitch. This struct is the reference. */
struct fb_fix_screeninfo {
	char id[16];
	unsigned long smem_start; /* physical address */
	uint32_t smem_len;
	uint32_t type;
	uint32_t type_aux;
	uint32_t visual;
	uint16_t xpanstep;
	uint16_t ypanstep;
	uint16_t ywrapstep;
	uint32_t line_length;
	unsigned long mmio_start; /* physical address */
	uint32_t mmio_len;
	uint32_t accel;
	uint16_t capabilities;
	uint16_t reserved[2];
};

#if __SIZEOF_LONG__ != 8
#error "fbprobe mirrors the LP64 (x86_64) fbdev ABI; see src/fb.rs for the same stance"
#endif
_Static_assert(sizeof(struct fb_var_screeninfo) == 160,
	       "fb_var_screeninfo does not match <linux/fb.h>");
_Static_assert(sizeof(struct fb_fix_screeninfo) == 80,
	       "fb_fix_screeninfo does not match <linux/fb.h> (LP64)");

static double now_sec(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void print_bitfield(const char *name, const struct fb_bitfield *b)
{
	printf("  %-14s offset=%2" PRIu32 " length=%2" PRIu32 " msb_right=%" PRIu32 "\n",
	       name, b->offset, b->length, b->msb_right);
}

static const char *fb_type_name(uint32_t t)
{
	switch (t) {
	case 0: return "PACKED_PIXEL";
	case 1: return "PLANES";
	case 2: return "INTERLEAVED_PLANES";
	case 3: return "TEXT";
	case 4: return "VGA_PLANES";
	default: return "?";
	}
}

static const char *fb_visual_name(uint32_t v)
{
	switch (v) {
	case 0: return "MONO01";
	case 1: return "MONO10";
	case 2: return "TRUECOLOR";
	case 3: return "PSEUDOCOLOR";
	case 4: return "DIRECTCOLOR";
	case 5: return "STATIC_PSEUDOCOLOR";
	default: return "?";
	}
}

int main(void)
{
	const char *path = "/dev/fb0";
	struct fb_var_screeninfo var;
	struct fb_fix_screeninfo fix;
	uint32_t *fb = MAP_FAILED;
	size_t bytes, nwords;
	double t0, t1, fill_s, read_s;
	size_t bad = 0;
	int fd, ret = 1;

	printf("fbprobe: opening %s\n", path);
	fd = open(path, O_RDWR);
	if (fd < 0) {
		printf("fbprobe: FAIL open %s: %s\n", path, strerror(errno));
		return 1;
	}
	printf("fbprobe: ok (fd %d)\n", fd);

	memset(&var, 0, sizeof(var));
	if (ioctl(fd, FBIOGET_VSCREENINFO, &var) != 0) {
		printf("fbprobe: FAIL FBIOGET_VSCREENINFO: %s\n", strerror(errno));
		goto out;
	}
	printf("fbprobe: FBIOGET_VSCREENINFO ok (sizeof=%zu)\n", sizeof(var));
	printf("  xres          =%" PRIu32 "\n", var.xres);
	printf("  yres          =%" PRIu32 "\n", var.yres);
	printf("  xres_virtual  =%" PRIu32 "\n", var.xres_virtual);
	printf("  yres_virtual  =%" PRIu32 "\n", var.yres_virtual);
	printf("  xoffset       =%" PRIu32 "\n", var.xoffset);
	printf("  yoffset       =%" PRIu32 "\n", var.yoffset);
	printf("  bits_per_pixel=%" PRIu32 "\n", var.bits_per_pixel);
	printf("  grayscale     =%" PRIu32 "\n", var.grayscale);
	print_bitfield("red", &var.red);
	print_bitfield("green", &var.green);
	print_bitfield("blue", &var.blue);
	print_bitfield("transp", &var.transp);
	printf("  nonstd        =%" PRIu32 "\n", var.nonstd);
	printf("  activate      =%" PRIu32 "\n", var.activate);
	printf("  height        =%" PRIu32 " mm\n", var.height);
	printf("  width         =%" PRIu32 " mm\n", var.width);
	printf("  accel_flags   =%" PRIu32 "\n", var.accel_flags);
	printf("  pixclock      =%" PRIu32 "\n", var.pixclock);
	printf("  left_margin   =%" PRIu32 "\n", var.left_margin);
	printf("  right_margin  =%" PRIu32 "\n", var.right_margin);
	printf("  upper_margin  =%" PRIu32 "\n", var.upper_margin);
	printf("  lower_margin  =%" PRIu32 "\n", var.lower_margin);
	printf("  hsync_len     =%" PRIu32 "\n", var.hsync_len);
	printf("  vsync_len     =%" PRIu32 "\n", var.vsync_len);
	printf("  sync          =%" PRIu32 "\n", var.sync);
	printf("  vmode         =%" PRIu32 "\n", var.vmode);
	printf("  rotate        =%" PRIu32 "\n", var.rotate);
	printf("  colorspace    =%" PRIu32 "\n", var.colorspace);
	printf("  reserved      =%" PRIu32 " %" PRIu32 " %" PRIu32 " %" PRIu32 "\n",
	       var.reserved[0], var.reserved[1], var.reserved[2], var.reserved[3]);

	memset(&fix, 0, sizeof(fix));
	if (ioctl(fd, FBIOGET_FSCREENINFO, &fix) != 0) {
		printf("fbprobe: FAIL FBIOGET_FSCREENINFO: %s\n", strerror(errno));
		goto out;
	}
	printf("fbprobe: FBIOGET_FSCREENINFO ok (sizeof=%zu)\n", sizeof(fix));
	{
		char idbuf[17];
		memcpy(idbuf, fix.id, 16);
		idbuf[16] = '\0';
		printf("  id            =\"%s\"\n", idbuf);
	}
	printf("  smem_start    =0x%016" PRIx64 "\n", (uint64_t)fix.smem_start);
	printf("  smem_len      =%" PRIu32 " (%.1f MiB)\n", fix.smem_len,
	       (double)fix.smem_len / 1048576.0);
	printf("  type          =%" PRIu32 " (%s)\n", fix.type, fb_type_name(fix.type));
	printf("  type_aux      =%" PRIu32 "\n", fix.type_aux);
	printf("  visual        =%" PRIu32 " (%s)\n", fix.visual, fb_visual_name(fix.visual));
	printf("  xpanstep      =%" PRIu16 "\n", fix.xpanstep);
	printf("  ypanstep      =%" PRIu16 "\n", fix.ypanstep);
	printf("  ywrapstep     =%" PRIu16 "\n", fix.ywrapstep);
	printf("  line_length   =%" PRIu32 "\n", fix.line_length);
	printf("  mmio_start    =0x%016" PRIx64 "\n", (uint64_t)fix.mmio_start);
	printf("  mmio_len      =%" PRIu32 "\n", fix.mmio_len);
	printf("  accel         =%" PRIu32 "\n", fix.accel);
	printf("  capabilities  =%" PRIu16 "\n", fix.capabilities);
	printf("  reserved      =%" PRIu16 " %" PRIu16 "\n",
	       fix.reserved[0], fix.reserved[1]);

	if (fix.smem_len == 0) {
		printf("fbprobe: FAIL smem_len is 0, nothing to map\n");
		goto out;
	}

	fb = mmap(NULL, fix.smem_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (fb == MAP_FAILED) {
		printf("fbprobe: FAIL mmap: %s\n", strerror(errno));
		goto out;
	}
	printf("fbprobe: mmap ok (%" PRIu32 " bytes, MAP_SHARED RW)\n", fix.smem_len);

	/* Whole-mapping fill, so the number is comparable with the kernel's
	 * own full-screen fill figure (~3026 MB/s, plan §4). */
	bytes = fix.smem_len - fix.smem_len % 4;
	nwords = bytes / 4;
	printf("fbprobe: filling %zu bytes with self-identifying gradient (word i = 0xFF000000|i)\n",
	       bytes);
	t0 = now_sec();
	for (size_t i = 0; i < nwords; i++)
		fb[i] = 0xFF000000u | (uint32_t)(i & 0x00FFFFFFu);
	t1 = now_sec();
	fill_s = t1 - t0;
	printf("fbprobe: fill %zu bytes in %.3f ms = %.0f MB/s\n",
	       bytes, fill_s * 1e3, (double)bytes / 1e6 / fill_s);

	/* Round-trip: read every word back and compare. This is the mmap/UC
	 * proof — a fill can hit WC buffers and lie, a read-back cannot. */
	t0 = now_sec();
	for (size_t i = 0; i < nwords; i++) {
		uint32_t want = 0xFF000000u | (uint32_t)(i & 0x00FFFFFFu);
		uint32_t got = fb[i];
		if (got != want) {
			if (bad == 0)
				printf("fbprobe: first mismatch at word %zu: want %08" PRIx32 " got %08" PRIx32 "\n",
				       i, want, got);
			bad++;
		}
	}
	t1 = now_sec();
	read_s = t1 - t0;
	printf("fbprobe: read-back %s: %zu/%zu words match (%.3f ms = %.0f MB/s read)\n",
	       bad == 0 ? "ok" : "FAIL", nwords - bad, nwords,
	       read_s * 1e3, (double)bytes / 1e6 / read_s);
	printf("fbprobe: samples: word[0]=%08x word[1]=%08x word[%zu]=%08x word[%zu]=%08x\n",
	       fb[0], fb[1], nwords / 2, fb[nwords / 2], nwords - 1, fb[nwords - 1]);

	if (bad == 0 && fill_s > 0.0)
		ret = 0;
	else if (fill_s <= 0.0)
		printf("fbprobe: FAIL fill timer did not advance\n");

out:
	if (fb != MAP_FAILED)
		munmap(fb, fix.smem_len);
	close(fd);
	printf("fbprobe: %s\n", ret == 0 ? "PASS" : "FAIL");
	return ret;
}

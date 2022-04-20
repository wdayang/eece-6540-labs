//==============================================================
// DPC++ Example
//
// Image rotation with DPC++
//
// Author: Yan Luo
//
// Copyright ©  2020-
//
// MIT License
//

#include <CL/sycl.hpp>
#include <array>
#include <math.h>
#include <iostream>
#include "dpc_common.hpp"
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;

// useful header files for image convolution
#include "utils.h"
#include "bmp-utils.h"
#include "gold.h"

using Duration = std::chrono::duration<double>;
class Timer
{
public:
  Timer() : start(std::chrono::steady_clock::now()) {}

  Duration elapsed()
  {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<Duration>(now - start);
  }

private:
  std::chrono::steady_clock::time_point start;
};

static const char *inputImagePath = "./Images/cat.bmp";

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)
#define DEVICE_NAME_LEN 128
static char dev_name[DEVICE_NAME_LEN];

//************************************
// Image Rotation in DPC++ on device:
//************************************
void ImageRotation(queue &q, void *image_in, void *image_out,
               size_t ImageRows, size_t ImageCols)
{
  // We create images for the input and output data.
  // Images objects are created from a host pointer together with provided
  // channel order and channel type.
  // image_in is a host side buffer of size ImageRows x ImageCols
  // each data item in image_in is float, representing a pixel
  // In the example file cat.bmp, each pixel is of 8-bit color, so we just
  // use "r" as channel order which replicates the value in all R component
  // in the image object
  // The channel type is set as fp32
  //
  image<2> srcImage(image_in, image_channel_order::r, image_channel_type::fp32,
                    range<2>(ImageCols, ImageRows));

  image<2> dstImage(image_out, image_channel_order::r, image_channel_type::fp32,
                    range<2>(ImageCols, ImageRows));

  // Create the range object for the pixel data managed by the image.
  range<2> num_items{ImageCols, ImageRows};

  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  q.submit([&](handler &h)
           {
      // Create an accessor to image with access permission: read, write or
      // read/write. The accessor is a way to access the memory in the image.
      // When accessing images, the accessor element type is used to specify 
      // how the image should be read from or written to. 
      // It can be either int4, uint4 or float4. 
      accessor<float4, 2, access::mode::read, access::target::image> srcPtr(
        srcImage, h);

      accessor<float4, 2, access::mode::write, access::target::image> dstPtr(
        dstImage, h);

      // Sampler are used to regulate access of the image
      sampler mysampler(coordinate_normalization_mode::unnormalized,
                    addressing_mode::clamp, filtering_mode::nearest);

      // rotate the image by an angle of
      float theta = 0.4;

      // Use parallel_for to run image convolution in parallel on device. This
      // executes the kernel.
      //    1st parameter is the number of work items.
      //    2nd parameter is the kernel, a lambda that specifies what to do per
      //    work item. The parameter of the lambda is the work item id.
      // DPC++ supports unnamed lambda kernel by default.
      h.parallel_for(num_items, [=](id<2> item) 
      { 

        // get row and col of the pixel assigned to this work item
        int source_x = item[0];
        int source_y = item[1];

        // DPC++ Coordinate objects for source and desintation locations
        int2 source_coords;
        int2 destination_coords;

        // Set source coordinates 
        source_coords[0] = source_x;
        source_coords[1] = source_y;

        // Initalize output pixel
        float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};

        // Get source pixel and assign value to sum
        float4 pixel = srcPtr.read(source_coords, mysampler);
        sum[0] = pixel[0];

        // Calculate new position of pixel
        float destination_x = cos(theta)*source_x - sin(theta)*source_y;
        float destination_y = sin(theta)*source_x + cos(theta)*source_y;

        // Set destination coordinates 
        destination_coords[0] = int(destination_x);
        destination_coords[1] = int(destination_y);

        // Range checking
        if (destination_coords[0] >= 0 && destination_coords[0] < ImageCols &&
            destination_coords[1] >= 0 && destination_coords[1] < ImageRows){
              dstPtr.write(destination_coords, sum);
          }
      }
    ); });
}

int main()
{
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  ext::intel::fpga_emulator_selector d_selector;
#elif FPGA
  // DPC++ extension: FPGA selector on systems with FPGA card.
  ext::intel::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
  // default_selector d_selector;
  cpu_selector d_selector;
#endif

  float *hInputImage;
  float *hOutputImage;

  int imageRows;
  int imageCols;
  int i;

  /* Read in the BMP image */
  hInputImage = readBmpFloat(inputImagePath, &imageRows, &imageCols);
  printf("imageRows=%d, imageCols=%d\n", imageRows, imageCols);
  /* Allocate space for the output image */
  hOutputImage = (float *)malloc(imageRows * imageCols * sizeof(float));
  for (i = 0; i < imageRows * imageCols; i++)
    hOutputImage[i] = 0.0;

  Timer t;

  try
  {
    queue q(d_selector, dpc_common::exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";

    // Image Rotation in DPC++
    ImageRotation
  (q, hInputImage, hOutputImage, imageRows, imageCols);
  }
  catch (exception const &e)
  {
    std::cout << "An exception is caught for image rotation.\n";
    std::terminate();
  }
  std::cout << t.elapsed().count() << " seconds\n";

  /* Save the output bmp */
  printf("Successful! the output image is saved as: cat-rotated.bmp\n");
  writeBmpFloat(hOutputImage, "cat-rotated.bmp", imageRows, imageCols,
                inputImagePath);

  return 0;
}

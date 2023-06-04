#include <CL/sycl.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace sycl;

static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const& e : e_list) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const& e) {
      std::cout << "General error" << std::endl;
      std::terminate();
    }
  }
};

int main(int argc, char* argv[]) {
  int numOfIntervals = std::stoi(argv[1]);
  std::string deviceType = static_cast<std::string>(argv[2]);

  device dev;
  if (deviceType == "gpu") {
    std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
    std::vector<sycl::device> devices = platforms[3].get_devices();
    dev = devices[0];
  } else {
    if (deviceType == "cpu") device(cpu_selector{});
  }

  std::cout << "Number of rectangles: " << numOfIntervals << " x "
            << numOfIntervals << std::endl;
  std::cout << "Target device: " << dev.get_info<sycl::info::device::name>()
            << std::endl;

  const int groupSize = 16;
  const int groupNum = 16;
  std::vector<float> resultBuf(groupNum * groupNum, 0.0f);
  property_list props{sycl::property::queue::enable_profiling()};
  queue queue(dev, exception_handler, props);
  try {
    buffer<float, 1> buf_a(resultBuf.data(), resultBuf.size());
    sycl::event event = queue.submit([&](handler& cgh) {
      auto out_a = buf_a.get_access<sycl::access::mode::write>(cgh);

      cgh.parallel_for(
          nd_range<2>(range<2>(groupNum * groupSize, groupNum * groupSize),
                      range<2>(groupSize, groupSize)),
          [=](nd_item<2> item) {
            const float x = static_cast<float>(item.get_global_id(0) + 0.5) /
                            numOfIntervals;
            const float y = static_cast<float>(item.get_global_id(1) + 0.5) /
                            numOfIntervals;
            const float add =
                static_cast<float>(item.get_global_range(0)) / numOfIntervals;
            float res = 0;
            for (float _x = x; _x <= 1; _x += add) {
              for (float _y = y; _y <= 1; _y += add) {
                res += sin(_x) * cos(_y) / numOfIntervals / numOfIntervals;
              }
            }
            float reducedRes =
                reduce_over_group(item.get_group(), res, std::plus<float>());
            if (item.get_local_id(0) == 0 && item.get_local_id(1) == 0) {
              out_a[item.get_group(0) + item.get_group(1) * groupNum] =
                  reducedRes;
            }
          });
    });
    queue.wait();

    uint64_t start =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    uint64_t end =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();

    std::cout << "Kernel execution time: "
              << static_cast<float>(end - start) / 1000000 << " ms"
              << std::endl;
  } catch (sycl::exception& ex) {
    std::cout << "Synchronous error: " << ex.what() << std::endl;
  }

  float expected = 2 * sin(1.0f / 2) * sin(1.0f / 2) * sin(1);
  float result = 0.0f;
  for (int i = 0; i < resultBuf.size(); i++) result += resultBuf[i];
  float diff = std::fabs(expected - result);
  std::cout << "Expected: " << expected << std::endl;
  std::cout << "Computed: " << result << std::endl;
  std::cout << "Difference: " << diff << std::endl;

  return 0;
}
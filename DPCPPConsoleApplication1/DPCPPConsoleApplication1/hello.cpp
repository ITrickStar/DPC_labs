#include <CL/sycl.hpp>
#include <ext/intel/fpga_extensions.hpp>
#include <iostream>
#include <vector>
using namespace sycl;

int main(int argc, char* argv[]) {
  std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
  int platformIdx = 0;
  for (auto platform : platforms) {
    std::cout << "Platform #" << platformIdx << ": "
              << platform.get_info<sycl::info::platform::name>() << std::endl;
    std::vector<sycl::device> devices = platform.get_devices();
    int deviceIdx = 0;
    for (auto device : devices) {
      std::cout << "-- Device #" << deviceIdx << ": "
                << device.get_info<sycl::info::device::name>() << std::endl;
    }
    platformIdx++;
  }

  std::cout << std::endl;

  platformIdx = 0;
  for (auto platform : platforms) {
    std::vector<sycl::device> devices = platform.get_devices();
    int deviceIdx = 0;
    for (auto device : devices) {
      std::cout << device.get_info<sycl::info::device::name>() << std::endl;
      {
        buffer<int> platformIdBuf(&platformIdx, 1);
        buffer<int> deviceIdBuf(&deviceIdx, 1);
        queue queue(device);
        queue.submit([&](handler& cgh) {
          auto pId = platformIdBuf.get_access<access::mode::read>(cgh);
          auto dId = deviceIdBuf.get_access<access::mode::read>(cgh);

          sycl::stream s(1024, 80, cgh);
          cgh.parallel_for(4, [=](int i) {
            s << "[" << i << "] Hello from platform " << pId[0]
              << " and device " << dId[0] << sycl::endl;
          });
        });
        queue.wait();
      }
    }
    platformIdx++;
  }
  return 0;
}

#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;

static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const& e : e_list) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const& e) {
      std::cout << "General error" << std::endl;
    }
  }
};

float getRandom() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

std::vector<float> initVector(int N) {
  std::vector<float> result;
  for (int i = 0; i < N; i++) {
    result.push_back(getRandom() * 2 - 1);
  }
  return result;
}

std::vector<float> initMatrix(int N) {
  std::vector<float> result;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      result.push_back(getRandom() * 2 - 1);
      if (i == j) {
        result[result.size() - 1] = N * 2;
      }
    }
  }
  return result;
}

float getNorm(int N, float* x) {
  float result = fabs(x[0]);
  for (int i = 1; i < N; i++) {
    if (result < fabs(x[i])) {
      result = fabs(x[i]);
    }
  }
  return result;
}

float getAchivedAccuracy(std::vector<float> A, std::vector<float> b,
                         std::vector<float> x) {
  std::vector<float> diff;
  for (int i = 0; i < b.size(); i++) {
    float sum = 0;
    for (int j = 0; j < b.size(); j++) {
      sum += A[j * b.size() + i] * x[j];
    }
    diff.push_back(sum - b[i]);
  }
  return getNorm(b.size(), &diff[0]);
}

void computeOnAccessors(sycl::device device, int N, std::vector<float> A,
                        std::vector<float> b, float accuracy, int maxIt) {
  property_list props{property::queue::enable_profiling()};
  queue queue(device, exception_handler, props);
  size_t time = 0;
  std::vector<float> x1(N, 0.0f);
  std::vector<float> x2(N, 0.0f);
  std::vector<float> diff(N, 0.0f);
  {
    buffer<float> buf_A(A.data(), A.size());
    buffer<float> buf_b(b.data(), b.size());
    buffer<float> buf_x1(x1.data(), x1.size());
    buffer<float> buf_x2(x2.data(), x2.size());
    auto buff_xk = &buf_x1;
    auto buff_xkp1 = &buf_x2;
    buffer<float> buf_diff(diff.data(), diff.size());
    for (int it = 0; it < maxIt; it++) {
      event e = queue.submit([&](handler& cgh) {
        auto A = buf_A.get_access<access::mode::read>(cgh);
        auto b = buf_b.get_access<access::mode::read>(cgh);
        auto x1 = buff_xk->get_access<access::mode::read>(cgh);
        auto x2 = buff_xkp1->get_access<access::mode::write>(cgh);
        auto diff = buf_diff.get_access<access::mode::write>(cgh);

        cgh.parallel_for(range<1>(N), [=](id<1> item) {
          x2[item] = b[item];
          for (int j = 0; j < N; j++) {
            x2[item] -= A[item + j * N] * x1[j];
          }
          x2[item] += A[item + item * N] * x1[item];
          x2[item] /= A[item + item * N];
          diff[item] = x1[item] - x2[item];
        });
      });
      e.wait();

      auto start = e.get_profiling_info<info::event_profiling::command_start>();
      auto end = e.get_profiling_info<info::event_profiling::command_end>();
      time += end - start;
      auto tmp = buff_xk;
      buff_xk = buff_xkp1;
      buff_xkp1 = tmp;
      auto host_diff = buf_diff.get_host_access();
      auto host_x1 = buff_xk->get_host_access();
      if (getNorm(N, host_diff.get_pointer()) /
              getNorm(N, host_x1.get_pointer()) <
          accuracy) {
        break;
      }
    }
  }
  std::cout << "[Accessors] Time: " << static_cast<float>(time) / 1000000
            << " ms Accuracy: " << getAchivedAccuracy(A, b, x1) << std::endl;
}

void computeOnShared(sycl::device device, int N, std::vector<float> A,
                     std::vector<float> b, float accuracy, int maxIt) {
  property_list props{property::queue::enable_profiling()};
  queue queue(device, exception_handler, props);

  float* shared_A = sycl::malloc_shared<float>(N * N, queue);
  float* shared_b = sycl::malloc_shared<float>(N, queue);
  float* shared_xk = sycl::malloc_shared<float>(N, queue);
  float* shared_xkPlus = sycl::malloc_shared<float>(N, queue);
  float* shared_diff = sycl::malloc_shared<float>(N, queue);

  size_t time = 0;

  for (int i = 0; i < N; i++) {
    shared_b[i] = b[i];
    shared_xk[i] = 0;
    for (int j = 0; j < N; j++) {
      shared_A[i + j * N] = A[i + j * N];
    }
  }
  for (int it = 0; it < maxIt; it++) {
    sycl::event e = queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
      shared_xkPlus[i] = shared_b[i];
      for (int j = 0; j < N; j++) {
        shared_xkPlus[i] -= shared_A[i + j * N] * shared_xk[j];
      }
      shared_xkPlus[i] += shared_A[i + i * N] * shared_xk[i];
      shared_xkPlus[i] /= shared_A[i + i * N];
      shared_diff[i] = shared_xk[i] - shared_xkPlus[i];
    });
    e.wait_and_throw();
    auto start = e.get_profiling_info<info::event_profiling::command_start>();
    auto end = e.get_profiling_info<info::event_profiling::command_end>();
    time += end - start;

    float* tmp = shared_xk;
    shared_xk = shared_xkPlus;
    shared_xkPlus = tmp;
    if (getNorm(N, shared_diff) / getNorm(N, shared_xkPlus) < accuracy) {
      break;
    }
  }
  std::vector<float> x(N, 0.0);
  for (int i = 0; i < N; i++) {
    x[i] = shared_xk[i];
  }
  std::cout << "[Shared]    Time: " << static_cast<float>(time) / 1000000
            << " ms Accuracy: " << getAchivedAccuracy(A, b, x) << std::endl;

  sycl::free(shared_A, queue);
  sycl::free(shared_b, queue);
  sycl::free(shared_xk, queue);
  sycl::free(shared_xkPlus, queue);
  sycl::free(shared_diff, queue);
}

void computeOnDevice(sycl::device device, int N, std::vector<float> A,
                     std::vector<float> b, float accuracy, int maxIt) {
  std::vector<float> x(N, 0.0f);
  std::vector<float> diff(N, 0.0f);
  property_list props{property::queue::enable_profiling()};
  queue queue(device, exception_handler, props);
  float* dev_A = sycl::malloc_device<float>(N * N, queue);
  float* dev_b = sycl::malloc_device<float>(N, queue);
  float* dev_xk = sycl::malloc_device<float>(N, queue);
  float* dev_xkPlus = sycl::malloc_device<float>(N, queue);
  float* dev_diff = sycl::malloc_device<float>(N, queue);

  size_t time = 0;
  queue.memcpy(dev_A, A.data(), N * N * sizeof(float)).wait();
  queue.memcpy(dev_b, b.data(), N * sizeof(float)).wait();

  for (int it = 0; it < maxIt; it++) {
    sycl::event e = queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
      dev_xkPlus[i] = dev_b[i];
      for (int j = 0; j < N; j++) {
        dev_xkPlus[i] -= dev_A[i + j * N] * dev_xk[j];
      }
      dev_xkPlus[i] += dev_A[i + i * N] * dev_xk[i];
      dev_xkPlus[i] /= dev_A[i + i * N];
      dev_diff[i] = dev_xk[i] - dev_xkPlus[i];
    });
    e.wait();
    auto start = e.get_profiling_info<info::event_profiling::command_start>();
    auto end = e.get_profiling_info<info::event_profiling::command_end>();
    time += end - start;

    queue.memcpy(diff.data(), dev_diff, N * sizeof(float)).wait();
    queue.memcpy(x.data(), dev_xk, N * sizeof(float)).wait();

    float* tmp = dev_xk;
    dev_xk = dev_xkPlus;
    dev_xkPlus = tmp;
    if (getNorm(N, &diff[0]) / getNorm(N, &x[0]) < accuracy) {
      break;
    }
  }

  std::cout << "[Device]    Time: " << static_cast<float>(time) / 1000000
            << " ms Accuracy: " << getAchivedAccuracy(A, b, x) << std::endl;
  sycl::free(dev_A, queue);
  sycl::free(dev_b, queue);
  sycl::free(dev_xk, queue);
  sycl::free(dev_xkPlus, queue);
  sycl::free(dev_diff, queue);
}

int main(int argc, char* argv[]) {
  int numOfEquations = std::stoi(argv[1]);
  float targetAccuracy = std::stof(argv[2]);
  int maxNumOfIterations = std::stoi(argv[3]);
  std::string deviceType = static_cast<std::string>(argv[4]);

  device dev;
  if (deviceType == "gpu") {
    std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
    std::vector<sycl::device> devices = platforms[3].get_devices();
    dev = devices[0];
  } else {
    if (deviceType == "cpu") device(cpu_selector{});
  }

  std::cout << "Target device: " << dev.get_info<sycl::info::device::name>()
            << std::endl;

  std::vector<float> A = initMatrix(numOfEquations);
  std::vector<float> b = initVector(numOfEquations);

  computeOnAccessors(dev, numOfEquations, A, b, targetAccuracy,
                     maxNumOfIterations);
  computeOnShared(dev, numOfEquations, A, b, targetAccuracy,
                  maxNumOfIterations);
  computeOnDevice(dev, numOfEquations, A, b, targetAccuracy,
                  maxNumOfIterations);

  return 0;
}

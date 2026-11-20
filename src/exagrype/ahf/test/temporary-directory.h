#ifndef APPARENT_HORIZON_FINDER_TEST_TEMPORARY_DIRECTORY_H_
#define APPARENT_HORIZON_FINDER_TEST_TEMPORARY_DIRECTORY_H_

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>

namespace exahype2::ccz4::ahf::tests {

// A new, uniquely named directory in the system temp folder, removed with its
// contents when the object goes out of scope. Tests that write files use it, so
// that concurrent runs (ctest -j, several checkouts, parallel CI jobs) and
// leftovers from earlier runs cannot affect each other. A failed assert aborts
// without running the destructor, which leaves the files for inspection.
class TemporaryDirectory {
public:
  explicit TemporaryDirectory(const std::string& prefix) {
    std::random_device device;
    std::mt19937_64 generator(
      (static_cast<std::uint64_t>(device()) << 32)
      ^ static_cast<std::uint64_t>(
          std::chrono::steady_clock::now().time_since_epoch().count()));
    for (int attempt = 0; attempt < 100; ++attempt) {
      const auto candidate = std::filesystem::temp_directory_path()
                             / (prefix + "-" + std::to_string(generator()));
      if (std::filesystem::create_directory(candidate)) {
        _path = candidate;
        return;
      }
    }
    throw std::runtime_error(
      "TemporaryDirectory: cannot create a unique directory for " + prefix);
  }

  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(_path, ignored);
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

  const std::filesystem::path& path() const { return _path; }

private:
  std::filesystem::path _path;
};

}  // namespace exahype2::ccz4::ahf::tests

#endif  // APPARENT_HORIZON_FINDER_TEST_TEMPORARY_DIRECTORY_H_

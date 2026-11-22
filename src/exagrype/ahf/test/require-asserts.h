#ifndef APPARENT_HORIZON_FINDER_TEST_REQUIRE_ASSERTS_H_
#define APPARENT_HORIZON_FINDER_TEST_REQUIRE_ASSERTS_H_

// The AHF tests check with assert, which NDEBUG compiles away: built with it,
// every test would pass whatever the code does. Every test program includes
// this header, so such a build fails instead. Build the tests in Debug.
#ifdef NDEBUG
#error "The AHF tests check with assert and NDEBUG removes it: build them without NDEBUG (CMAKE_BUILD_TYPE=Debug)"
#endif

#endif  // APPARENT_HORIZON_FINDER_TEST_REQUIRE_ASSERTS_H_

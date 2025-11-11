/*
Licensed under the MIT License.
Copyright 2025 Daniel Lidstrom
*/

#ifndef CONNECTFOUR_TESTS_H
#define CONNECTFOUR_TESTS_H

#include <string>
#include <vector>
#include <functional>

namespace ConnectFour {

struct TestResult {
  std::string name;
  bool passed;
  std::string message;
};

class TestRunner {
public:
  void AddTest(const std::string& name, std::function<TestResult()> test);
  void RunAll();
  int GetFailureCount() const { return failureCount; }

private:
  std::vector<std::pair<std::string, std::function<TestResult()>>> tests;
  int failureCount = 0;
};

// Test functions
TestResult TestBoardBasics();
TestResult TestMinimaxBasics();
TestResult TestMCTSBasics();
TestResult TestMCTSVsMinimax2Ply();
TestResult TestMCTSFindsWinInOne();
TestResult TestMCTSBlocksLossInOne();
TestResult TestUntrainedMCTSVsMinimax2();

} // namespace ConnectFour

#endif // CONNECTFOUR_TESTS_H

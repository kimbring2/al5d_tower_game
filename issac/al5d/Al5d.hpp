#pragma once

#include <memory>
#include <string>

#include "engine/alice/alice_codelet.hpp"

namespace isaac {

class Al5d : public alice::Codelet {
 public:
  void start() override;
  void tick() override;
  void stop() override;
  ISAAC_PARAM(std::string, message, "Hello AL5D!");
};

}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::Al5d);
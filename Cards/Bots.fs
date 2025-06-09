module Bots

open Domain

module HeuristicBot =
  let pickMove (state: GameState) (player: PlayerState) : Action =
    // To be implemented — simple rules like:
    // - Prefer lowest valid card
    // - Prefer burning with 10 or 4-of-a-kind
    // - Avoid picking up stack
    PickUpStack  // Placeholder

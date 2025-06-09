[<AutoOpen>]
module Prelude

let randFloat =
  let P = 2147483647u
  let A = 16807u;
  let mutable current = 1u
  let inner() =
    current <- current * A % P;
    let result = float current / float P
    result
  inner

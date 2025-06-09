module Domain

type Suit = Clubs | Diamonds | Hearts | Spades
type Rank = Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King | Ace

type Card = { Rank: Rank; Suit: Suit }

type PlayerState = {
  Hand: Card list
  KnownOpponentCards: Card list
  UnknownOpponentCount: int
}

type StackState = {
  Cards: Card list
  TopRank: Rank option   // None if stack is empty or reset
  TopCount: int          // Number of same-rank cards on top (up to 3)
}

type GameState = {
  Player1: PlayerState
  Player2: PlayerState
  Stack: StackState
  DrawPile: Card list
  CurrentPlayer: int  // 0 or 1
  BurnedPile: Card list
}

type Action =
  | Play of Card list
  | PickUpStack

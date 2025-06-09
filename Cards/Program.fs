module CardGame

open System

// --- Definitions ---

type Suit = Clubs | Diamonds | Hearts | Spades

type Rank =
  | Three | Four | Five | Six | Seven | Eight | Nine | Ten
  | Jack | Queen | King | Ace | Two

let allRanks = [ Three; Four; Five; Six; Seven; Eight; Nine; Ten; Jack; Queen; King; Ace; Two ]
let rankValue rank = allRanks |> List.findIndex ((=) rank)

[<StructuralEquality; StructuralComparison>]
type Card = { Rank: Rank; Suit: Suit }

let suitSymbol = function
  | Clubs -> "♣"
  | Diamonds -> "♦"
  | Hearts -> "♥"
  | Spades -> "♠"

let rankSymbol = function
  | Three -> "3"
  | Four -> "4"
  | Five -> "5"
  | Six -> "6"
  | Seven -> "7"
  | Eight -> "8"
  | Nine -> "9"
  | Ten -> "10"
  | Jack -> "J"
  | Queen -> "Q"
  | King -> "K"
  | Ace -> "A"
  | Two -> "2"

let formatCard (card: Card) =
  let symbol = rankSymbol card.Rank
  let suit = suitSymbol card.Suit
  sprintf "%s%s" symbol suit

let formatHand (hand: Card list) =
  hand |> List.map formatCard |> String.concat " "

// --- Game State ---

type PlayerState = {
  Hand: Card list
  KnownOpponentCards: Card list
  UnknownOpponentCount: int
}

type StackState = {
  Cards: Card list
  TopRank: Rank option
  TopCount: int
}

type GameState = {
  Player1: PlayerState
  Player2: PlayerState
  Stack: StackState
  DrawPile: Card list
  CurrentPlayer: int // 0 or 1
  BurnedPile: Card list
}

// --- Moves ---
type Action =
  | Play of Card list
  | PickUpStack

// --- Utilities ---

let shuffle (deck: Card list) : Card list =
  let rnd = Random()
  deck |> List.sortBy (fun _ -> rnd.Next())

let freshDeck : Card list =
  [ for s in [Clubs; Diamonds; Hearts; Spades] do
    for r in allRanks do
      yield { Rank = r; Suit = s } ]

// --- Game Mechanics ---

let drawCards count pile =
  let drawn = List.take count pile
  let remaining = List.skip count pile
  drawn, remaining

let refillHand hand pile =
  let missing = max 0 (3 - List.length hand)
  let drawn, rest = drawCards missing pile
  hand @ drawn, rest

let isValidPlay (cards: Card list) (stack: StackState) =
  match cards with
  | [] -> false
  | hd::_ ->
    let allSame = cards |> List.forall (fun c -> c.Rank = hd.Rank)
    if not allSame then false
    else
      match hd.Rank, stack.TopRank with
      | Two, _ -> true // 2 resets
      | Ten, _ -> true // 10 burns
      | r, Some top -> rankValue r >= rankValue top
      | _, None -> true

let updateStack (cards: Card list) (stack: StackState) =
  match cards with
  | [] -> stack
  | hd::_ ->
    let newCards = stack.Cards @ cards
    let newTopRank = Some hd.Rank
    let sameOnTop =
      newCards
      |> List.rev
      |> List.takeWhile (fun c -> c.Rank = hd.Rank)
      |> List.length
    { Cards = newCards; TopRank = newTopRank; TopCount = sameOnTop }

let burnStack = { Cards = []; TopRank = None; TopCount = 0 }

let applyMove (state: GameState) (action: Action) : GameState =
  let player, opponent = if state.CurrentPlayer = 0 then state.Player1, state.Player2 else state.Player2, state.Player1
  match action with
  | PickUpStack ->
    let newHand = player.Hand @ state.Stack.Cards
    let newPlayer = { player with Hand = newHand }
    let stack = burnStack
    let p1, p2 = if state.CurrentPlayer = 0 then newPlayer, opponent else opponent, newPlayer
    {
      state with
        Player1 = p1; Player2 = p2;
        Stack = stack;
        CurrentPlayer = 1 - state.CurrentPlayer
    }
  | Play cards ->
    let remainingHand = player.Hand |> List.except cards
    let stack' = updateStack cards state.Stack
    let burn =
      match cards with
      | { Rank = Ten }::_ -> true
      | { Rank = r }::_ when stack'.TopCount = 4 -> true
      | _ -> false
    let newStack, burned = if burn then burnStack, stack'.Cards else stack', []
    let newHand, newPile = refillHand remainingHand state.DrawPile
    let newPlayer = { player with Hand = newHand }
    let p1, p2 = if state.CurrentPlayer = 0 then newPlayer, opponent else opponent, newPlayer
    let samePlayer =
      match cards with
      | { Rank = Two }::_ -> true
      | { Rank = Ten }::_ -> true
      | { Rank = r }::_ when stack'.TopCount = 4 -> true
      | _ -> false
    {
      state with
        Player1 = p1; Player2 = p2;
        Stack = newStack;
        BurnedPile = burned @ state.BurnedPile;
        DrawPile = newPile;
        CurrentPlayer = if samePlayer then state.CurrentPlayer else 1 - state.CurrentPlayer
    }

let getValidMoves (player: PlayerState) (stack: StackState) : Action list =
  let grouped =
    player.Hand
    |> List.groupBy (fun c -> c.Rank)
    |> List.map snd
    |> List.filter (fun g -> isValidPlay g stack)
  let plays = grouped |> List.map Play
  if plays = [] then [ PickUpStack ] else plays

// --- Initialization ---

let deal deck =
  let p1, rest1 = drawCards 3 deck
  let p2, rest2 = drawCards 3 rest1
  p1, p2, rest2

let initialState () =
  let deck = shuffle freshDeck
  let p1, p2, rest = deal deck
  {
    Player1 = { Hand = p1; KnownOpponentCards = []; UnknownOpponentCount = 3 }
    Player2 = { Hand = p2; KnownOpponentCards = []; UnknownOpponentCount = 3 }
    Stack = burnStack
    DrawPile = rest
    CurrentPlayer = 0
    BurnedPile = []
  }

// --- Interactive Loop ---

let rec gameLoop (state: GameState) =
  let player = if state.CurrentPlayer = 0 then state.Player1 else state.Player2
  let name = if state.CurrentPlayer = 0 then "Player 1" else "Player 2"
  printfn "\n--- %s's Turn ---" name
  printfn "Hand: %s" (formatHand player.Hand)
  printfn "Stack: %s" (formatHand state.Stack.Cards)

  let moves = getValidMoves player state.Stack
  moves |> List.iteri (fun i m ->
    match m with
    | Play cards -> printfn "%d: Play %s" i (formatHand cards)
    | PickUpStack -> printfn "%d: Pick up stack" i)

  printf "Choose move: "
  let choice = Console.ReadLine() |> int
  let move = moves[choice]
  let newState = applyMove state move
  if newState.Player1.Hand = [] && newState.DrawPile = [] then
    printfn "Player 1 wins!"
  elif newState.Player2.Hand = [] && newState.DrawPile = [] then
    printfn "Player 2 wins!"
  else
    gameLoop newState

// Entry point
[<EntryPoint>]
let main _ =
  let state = initialState()
  gameLoop state
  0

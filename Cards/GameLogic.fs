module GameLogic

open Domain

let isResetCard (card: Card) =
  card.Rank = Two

let isBurnCard (cards: Card list) =
  cards.Length = 1 && cards.Head.Rank = Ten
  || cards |> List.map (fun c -> c.Rank) |> Set.ofList |> Set.count = 1 && cards.Length = 4

let canPlayOnTop (card: Card) (top: Rank option) =
  match top with
  | None -> true
  | Some topRank -> card.Rank >= topRank || isResetCard card

let groupByRank (hand: Card list) =
  hand
  |> List.groupBy (fun c -> c.Rank)
  |> List.map (fun (rank, cards) -> (rank, cards))

let getValidMoves (hand: Card list) (top: Rank option) : Card list list =
  groupByRank hand
  |> List.collect (fun (_, cards) ->
    let playable = cards |> List.filter (fun c -> canPlayOnTop c top)
    match playable.Length with
    | 0 -> []
    | n ->
      // Generate all combinations from 1 to n cards of same rank
      [1..n]
      |> List.map (fun count -> playable |> List.truncate count)
  )

let applyMove (game: GameState) (action: Action) : GameState =
  let current, _ =
    if game.CurrentPlayer = 0 then game.Player1, game.Player2
    else game.Player2, game.Player1

  match action with
  | PickUpStack ->
      let newHand = current.Hand @ game.Stack.Cards
      let updatedPlayer = { current with Hand = newHand }
      let nextState = if game.CurrentPlayer = 0 then
                        { game with Player1 = updatedPlayer; CurrentPlayer = 1 }
                      else
                        { game with Player2 = updatedPlayer; CurrentPlayer = 0 }
      { nextState with Stack = { game.Stack with Cards = []; TopRank = None; TopCount = 0 } }

  | Play cards ->
      let newHand = current.Hand |> List.except cards
      let newStack = game.Stack.Cards @ cards
      let playedRank = cards.Head.Rank
      let topCount =
        let recentSame =
          List.rev newStack
          |> List.takeWhile (fun c -> c.Rank = playedRank)
        recentSame.Length

      let burned =
        isBurnCard cards || topCount = 4

      let stack =
        if burned then
          { Cards = []; TopRank = None; TopCount = 0 }
        elif cards.Head.Rank = Two then
          { Cards = newStack; TopRank = None; TopCount = 0 }
        else
          { Cards = newStack; TopRank = Some playedRank; TopCount = topCount }

      let updatedPlayer = { current with Hand = newHand }

      // Refill hand from draw pile
      let handSize = newHand.Length
      let cardsToDraw = min (3 - handSize) (List.length game.DrawPile)
      let drawn, remainingDraw = game.DrawPile |> List.splitAt cardsToDraw
      let finalHand = newHand @ drawn
      let updatedPlayer = { updatedPlayer with Hand = finalHand }

      let newGame =
        if game.CurrentPlayer = 0 then
          { game with Player1 = updatedPlayer; DrawPile = remainingDraw }
        else
          { game with Player2 = updatedPlayer; DrawPile = remainingDraw }

      let nextPlayer =
        if burned || cards.Head.Rank = Ten || topCount = 4 then game.CurrentPlayer
        else 1 - game.CurrentPlayer

      { newGame with Stack = stack; CurrentPlayer = nextPlayer }

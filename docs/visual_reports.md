# Static player, visual test
## 14/04 ~13:00, rob PC

Network: obs, 64, 32, 20, act
RAY_BATCHES = 3
EPS_NUM = 10
MAX_CYCLES = 400

The network seems to learn that a certain position is good.
So at every next episode, enemies go back to that position, rather than trying to reach the player.

It's not like enemies learn to be close to themselves, it's that they all learn to go to the "position".

In this kind of test, there's no actual way for them to understand which is more important, between the player position and a specific self.pos.
Moreover, since they all do the same things, the others.pos tend to 0, giving them the impression that that
is also a good sign.

What we are seeing is them being confused by the first few episodes and then having to re-explore.

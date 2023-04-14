# Evasive player
## 14/04 ~13:20, rob PC

Network: obs, 64, 32, 20, act
RAY_BATCHES = 3
EPS_NUM = 10
MAX_CYCLES = 2500

Enemies do manage to understand they have to go towards the player.
But they seem to be slow at it. Although they do also learn to put themselves in ways that force the player into being static against a wall or corner.

When the player evades away (which usually happens after a collision or very close to that), enemies take their time to follow it.

Notice that right now every enemy is rewarded separately.



# Static player
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

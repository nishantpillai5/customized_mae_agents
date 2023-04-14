# Random player
## 14/04 ~15:30, rob PC

- Network: obs, 64, 32, 20, act
- RAY_BATCHES = 3
- EPS_NUM = 10
- MAX_CYCLES = 2500

Enemies do seem to spread out a bit when they have the player around them, since their reward is the same even if just one is close to the player, and can be higher later if they stay spread.

However in general they still stay all together.
One possible explanation for that could be found if we notice what happens in the part of the experiment where they do stay away from each others: the start.

At the start, enemies go around outside of the field, taking negative score, just because they are exploring. That is the moment where others.pos are higher (they are relative).
Well, so they are learning a negative scored associated to being away from each others.

Episode averages started around -.8, and then quickly stabilized around -.15



# Evasive player
## 14/04 ~13:40, rob PC

- Network: obs, 64, 32, 20, act
- RAY_BATCHES = 3
- EPS_NUM = 10
- MAX_CYCLES = 2500

This time reward is shared, and at the end of an episode the average reward is shown.
This average is basically the average distance with a bonus for every collision.

So, in this test the enemies seemed to be more spreadout, but still basically all in the same place.
The average reward was initially -0.85 to -0.55, and then stabilized around -0.3.
After a while I feel the average grew to -0.25.
The last episodes of the 3 instances scored: -0.26, -0.20 and -0.23

There is definitely an increase over time


# Evasive player
## 14/04 ~13:20, rob PC

- Network: obs, 64, 32, 20, act
- RAY_BATCHES = 3
- EPS_NUM = 10
- MAX_CYCLES = 2500

Enemies do manage to understand they have to go towards the player.
But they seem to be slow at it. Although they do also learn to put themselves in ways that force the player into being static against a wall or corner.

When the player evades away (which usually happens after a collision or very close to that), enemies take their time to follow it.

Notice that right now every enemy is rewarded separately.



# Static player
## 14/04 ~13:00, rob PC

- Network: obs, 64, 32, 20, act
- RAY_BATCHES = 3
- EPS_NUM = 10
- MAX_CYCLES = 400

The network seems to learn that a certain position is good.
So at every next episode, enemies go back to that position, rather than trying to reach the player.

It's not like enemies learn to be close to themselves, it's that they all learn to go to the "position".

In this kind of test, there's no actual way for them to understand which is more important, between the player position and a specific self.pos.
Moreover, since they all do the same things, the others.pos tend to 0, giving them the impression that that
is also a good sign.

What we are seeing is them being confused by the first few episodes and then having to re-explore.

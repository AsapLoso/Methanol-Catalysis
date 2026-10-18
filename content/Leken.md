---
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# In plain language

We want to turn acetylene into ethylene — an important industrial reaction — without going one
step too far and turning it into ethane.

The problem is that the same catalyst does both, and the second step is easy to overshoot. In
practice the reaction is controlled by adjusting conditions: temperature, pressure, how long the
gas spends in the reactor.

We are asking a different question: **can we control it better by varying the feed over time?**
If the inlet concentration of the reactants pulses instead of holding steady, the catalyst
surface never settles into one fixed state — and because the two reaction steps depend on the
surface differently, a pulsing feed can favour the one we want.

To find out which pulses work, we first need a model. This work builds one: a description of a
packed-bed reactor that tracks the gas along the tube, the gas inside the catalyst pellets, and
the molecules sitting on the catalyst surface. It is built from parameters published in the
literature rather than guessed, and every modelling simplification is written down next to the
reason for it.

The model is not finished being tested against experiments, and the search for the best input
profile has not started. What exists is the machinery to ask the question properly.

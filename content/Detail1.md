# About reactions

The chemical reaction we're interested in is acetylene semi-hydrogenation to ethylene:


$$C2H2 + H2 -> C2H4$$


The problem is that the reaction doesn't stop at ethylene, but it also produces ethane:

$$C2H2 + 2H2 -> C2H6$$

We want to maximize the production of ethylene and minimize the production of ethane.

And keep the total production high, besides the selectivity.

The ideal catalyst would be one that selectively hydrogenates acetylene to ethylene and then stops.

In reality, we can control the selectivity by adjusting the reaction conditions (temperature, pressure, residence time, etc.). This is grounded in x papers (the ones who talk about transients).

In this paper we're interested in manipulating the reation through **transient concentration profiles** by pulsing the concentrations of reactants.

By modelling the reaction using a rate-equation model (grounded in the papers on microkinetic parameters) we can then use control theory to design optimal **Input Profiles** (in this case H2 and C2H2 concentrations over time). 

By finding the optimal input profile we can then use it to control the reaction in a lab-scale setup (plug flow reactor). This is grounded in another set of papers.

To achieve this, we first need a model of the reaction. 

The reaction network consists of adsorption, surface reaction, and desorption steps (using $*$ to denote an adsorbed species, with vacant site fraction $\theta_* = 1 - \theta_{\mathrm{H}} - \theta_{\mathrm{C_2H_2}} - \theta_{\mathrm{C_2H_4}}$):

1. **Dissociative adsorption of $\mathrm{H_2}$:**
   $$\mathrm{H_2 + 2* \rightleftharpoons 2H*}$$
   $$R_1 = k_{1,\mathrm{ads}} P_{\mathrm{H_2}} \theta_*^2 - k_{1,\mathrm{des}} \theta_{\mathrm{H}}^2$$

2. **Molecular adsorption of $\mathrm{C_2H_2}$:**
   $$\mathrm{C_2H_2 + * \rightleftharpoons C_2H_2*}$$
   $$R_2 = k_{2,\mathrm{ads}} P_{\mathrm{C_2H_2}} \theta_* - k_{2,\mathrm{des}} \theta_{\mathrm{C_2H_2}}$$

3. **Surface hydrogenation to ethylene (reversible):**
   $$\mathrm{C_2H_2* + 2H* \rightleftharpoons C_2H_4* + 2*}$$
   $$R_3 = k_{3,\mathrm{fwd}} \theta_{\mathrm{C_2H_2}} \theta_{\mathrm{H}}^2 - k_{3,\mathrm{rev}} \theta_{\mathrm{C_2H_4}} \theta_*^2$$

4. **Desorption of target product (ethylene):**
   $$\mathrm{C_2H_4* \rightleftharpoons C_2H_4 + *}$$
   $$R_4 = k_{4,\mathrm{des}} \theta_{\mathrm{C_2H_4}} - k_{4,\mathrm{ads}} P_{\mathrm{C_2H_4}} \theta_*$$

5. **Over-hydrogenation to ethane (direct desorption to gas phase, irreversible):**
   $$\mathrm{C_2H_4* + 2H* \rightarrow C_2H_6(g) + 3*}$$
   $$R_5 = k_{5} \theta_{\mathrm{C_2H_4}} \theta_{\mathrm{H}}^2$$

Anyhow

We have the reaction. But it can only react on the catalyst surface, not in the gas phase.

So for our model we need to consider the gas-phase species, the adsorbed species and the catalyst surface as a whole. 

We do this through the conservation laws. Paper 2008 Berger (1D dynamic axial model): \

$$\frac{\partial C_i}{\partial t} = -\frac{u_s}{\varepsilon_b} \frac{\partial C_i}{\partial z} + D_{\mathrm{ax}} \frac{\partial^2 C_i}{\partial z^2} + \frac{\rho_s (1 - \varepsilon_b) C_t}{\varepsilon_b} R_i$$

It has accumulation ($\partial C_i/\partial t$), convective transport ($- \frac{u_s}{\varepsilon_b}\frac{\partial C_i}{\partial z}$), axial dispersion ($D_{\mathrm{ax}}\frac{\partial^2 C_i}{\partial z^2}$, negligible under plug-flow criteria $Pe_{\mathrm{ax}} > 100$), and catalytic surface generation/consumption ($R_i$).

[]
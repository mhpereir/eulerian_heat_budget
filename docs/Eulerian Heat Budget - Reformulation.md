------

Starting with the Eulerian form of heat transport in the infinitesimal form:
$$
\frac{\partial T}{\partial t} = - \vec{u}\cdot\vec\nabla_H T -\omega\frac{\partial T}{\partial P} + \omega\frac{RT}{c_P P} + \frac{J}{c_P},
$$
which we can re-write in a compact form, under assumption of divergent-less flow $(\vec \nabla \cdot \vec U =0)$:
$$
\frac{\partial T}{\partial t} = - \vec\nabla \cdot (\vec UT) + \omega\frac{RT}{c_P P} + \frac{J}{c_P}
$$
Recall also the Reynolds Transport Theorem:
$$
\begin{align*}
\Gamma(t) &= \int_{V(t)}\gamma(t)dV \\
\frac{d}{dt}\Gamma(t) &= \int_{V(t)}\frac{d\gamma}{dt} dV + \iint_A \gamma (\vec v_{b}\cdot\hat n) dA.
\end{align*}
$$
If we let $\gamma\equiv T$, we obtain,
$$
\frac{d}{dt}\int_{V(t)}TdV = \int_{V(t)} \frac{\partial T}{\partial t} dV + \iint_A T (\vec v_b \cdot \hat n) dA.
$$
For our problem, the boundary ($v_b$) is only non-zero at the surface:
$$
\frac{d}{dt}\int_{V(t)}TdV = \int_{V(t)} \frac{\partial T}{\partial t} dV + \iint_{\text{surface}} T (\vec v_b \cdot \hat n) dA.
$$


From here, we insert the definition of $\partial T/\partial t$ from the Eulerian form of heat transport:
$$
\frac{d}{dt}\int_{V(t)}TdV = \int_{V(t)}\bigg[ \vec \nabla \cdot(\vec U T) + \omega\frac{RT}{c_P P} + \frac{J}{c_P}\bigg]dv + \iint_{s} T (\vec v_{b}\cdot\hat n) dA
$$
Applying Gauss' Law:
$$
\frac{d}{dt}\int_{V(t)}TdV = -\iint_A T(\vec U \cdot \hat n)dA + \int_{V(t)}\bigg[\omega\frac{RT}{c_P P} + \frac{J}{c_P}\bigg]dv + \iint_s T (\vec v_{b}\cdot\hat n) dA
$$
If we expand the surface integral of the advection components:
$$
-\iint_A T(\vec U \cdot \hat n) dA = -\iint_{top+west+east+north+south}T(\vec U \cdot \hat n) dA - \iint_{\text{s}} T(\vec U \cdot \hat n) dA.
$$
At the surface, there is no transfer of mass across the boundary, so $\vec U \cdot \hat n\bigg|_{surface}= \omega =v_b$. 

So the surface "advection" term and the extra term from the Reynolds Transport theorem cancel!, yielding:
$$
\frac{d}{dt}\int_{V(t)} TdV = -\iint_{T+W+E+N+S} T(\vec U\cdot \hat n)dA +\int_{V(t)}\bigg[\omega\frac{RT}{c_P P} + \frac{J}{c_P}\bigg]dv
$$

### Mass Balance

In theory, mass-balance should be conserved
$$
\delta \mathcal{M} \equiv \iint (\vec U\cdot \hat n) dA + \frac{dV}{dt} = 0
$$
### Simplification

Defining a volume-averaged temperature
$$
\langle T \rangle \equiv \frac{1}{V(t)} \int_{V(t)} T dV
$$
about which we may define a spatial temperature anomaly,
$$
T'(t,\vec x) = T(t,\vec x) - \langle T\rangle (t).
$$

Replacing T with this anomaly expression for both the storage and advection terms, we obtain:
$$
\frac{d}{dt}\int_{V(t)} (\langle T \rangle +T')dV = -\iint_{T+W+E+N+S} (\langle T \rangle +T')(\vec U\cdot \hat n)dA +\int_{V(t)}\bigg[\omega\frac{RT}{c_P P} + \frac{J}{c_P}\bigg]dv
$$
where, at all times (t),
$$
\int_{V(t)} T' dV = 0
$$
so the LHS becomes:
$$
\frac{d}{dt}\int_{V(t)} (\langle T \rangle +T')dV = \langle T\rangle \frac{dV}{dt} + V \frac{d\langle T \rangle}{dt}
$$
and the advection term becomes:
$$
\iint (\langle T \rangle +T')(\vec U\cdot \hat n)dA = \langle T \rangle\iint(\vec U\cdot \hat n)dA + \iint T'(\vec U\cdot \hat n)dA
$$

Notice how the advection term carried within an offset based on the average temperature of the domain multiplied by the mass advection flux. From mass balance, we may replace the velocity flux term with the change in volume, yielding,
$$
\iint (\langle T \rangle +T')(\vec U\cdot \hat n)dA = -\langle T \rangle \frac{dV}{dt} + \iint T'(\vec U\cdot \hat n)dA
$$
Putting all of this together, we obtain:
$$
\frac{d\langle T\rangle}{dt} = -\iint_{T+W+E+N+S} T'(\vec U\cdot \hat n)dA +\int_{V(t)}\bigg[\omega\frac{RT}{c_P P} + \frac{J}{c_P}\bigg]dv
$$
So when the average temperature from the advection calculation, we must also change our term of the LHS.  
## Understanding uncertainty
### Mass Balance
In theory, mass-balance should be conserved throughout the time series:

$$
\delta \mathcal{M} \equiv \iint (\vec U\cdot \hat n) dA + \frac{dV}{dt} = 0
$$
Due to numerical resolution issues, this is rarely the case, so we can expect $\delta \mathcal{M}$ to be small non-zero:
$$\delta \mathcal{M}
\neq 0$$
### Temperature
The cancellation of the $\langle T \rangle$ terms in the previous simplification is based on the assumption of complete mass closure, which may not occur in practice due to numerical:
$$
\delta \mathcal{H} \equiv \langle T\rangle \bigg[\frac{dV}{dt} + \iint(\vec U\cdot\hat n)dA \bigg ] = \langle T\rangle \delta \mathcal{M}
$$

If we subtract $\langle T \rangle$ from the temperature field before calculating the advection component, we obtain the trivial result that $\delta \mathcal{H}=0$, which isn't entirely correct. 

In order to recover an estimate of uncertainty, we may use a different measure of temperature to scale the mass closure error:
$$
\delta \mathcal{H} \approx T_{scale} \delta \mathcal{M},
$$
where $T_{scale}$ is shown to be adequately described by the RMSE of $T'$;
$$
T_{scale} = \sqrt{\frac{1}{N} \sum_{t,i,j,k} (T_{t,i,j,k}-\langle T\rangle_{t})^2}
$$
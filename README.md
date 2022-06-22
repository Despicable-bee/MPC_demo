# Model Predictive Control (MPC) / Trajectory optimisation demo

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/spaceXRocket2.jpg"
    width="500px">

## Some brief history on MPC

MPC is a pretty broad term, as there are many generations that have been developed
    since the early 1960s. [This IJETT](http://ijettjournal.org/volume-4/issue-6/IJETT-V4I6P173.pdf)  article does a great job breaking it down the many
    generations. I highly encourage readers to have a look at it to get an 
    appreciation for how far modern control methods have come.

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/mpc_history_1.PNG"
    width="500px">
 
 ## So what will you be showing us today?

 It would be nice to show off some "*super advanced MPC example*" to demonstrate
    how cool I am, but it wouldn't be very informative from an engineering
    perspective.

Hence, today we shall look a little under the hood as to how the method 
    works, and (more importantly), what advancements in applied math / compsci
    allowed things like modern neural networks / Self-landing rockets to be 
    realised.

## The toy problem

I always find that an example, followed by a nice explanation is the best
way to learn things.

For the rest of this project, we will endeavour to optimally control the
    following dynamical system.

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/system_equation.PNG"
    width="200px">

Those who've done **chaos theory** will recognise this as a classic example
used to demonstrate the concept of regions of stability (along with the methods
    to demonstrate that a region is inherently stable, even though a path 
    through the system will likely be chaotic).

For those who don't know what I'm talking about, I've prepared a little script
to visualise the system in **x** and **y**.

Assuming you've install `numpy` and `matplotlib`, If you run: 

```
python3 main.py
```

After a few seconds you should see the following image:

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/toy_problem_vector_field.png"
    width="500px">

### Why does this system need to be controlled?
---
Now you're probably wondering "*why does this system need to be controlled?*",

If you look at the equation for rdot, you can probably see that as r^2 gets 
    closer to lambda (either from above or below), rdot will go to zero.
    
This (as you can see from the paths near the ring of stability) can take
    some time, as the gradient for r (rdot) near lambda, is pretty close to 
    zero.

That behaviour may not be optimal for our purposes (i.e. we may want to 
    construct a path from some point to the origin in the least amount of time
    possible).

A good analogy would be video game controls with an inadequate response time.

### How will this system be controlled?
---
The intuition behind how we will control this system is pretty straight forward.

We'll put my MS paint skills to the test with this one :)

We'll start off with a system at rest, with a starting condition at, say
x = 0.1, y = 0.1

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/how_1.PNG"
    width="500px">

Here, the **red circle** indicates the ring of stability, which at this point 
    inhabits a radius of **r = 3**.

The pink arrows are the direction of the gradient of the system at various 
    points.

The blue dot labelled **X0** denotes our starting point (i.e. X0 = (0.1, 0.1)).

At this stage, the point will fall to the ring of stability as per normal. 
    However, what if we were to increase the radius of the ring of stability 
    (i.e. **r >> 3**)?

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/how_2.PNG"
    width="500px">

The orange ring denotes the target position we're interested in 
    the point reaching.

The gradient is stronger the further the ring of stability is away, hence it should rise faster.

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/how_3.PNG"
    width="500px">

If we keep this **r >> 3** value going for enough timesteps, we should either 
    match or overshoot the desired radius. At this point, we can make the ring of stability smaller, in order to bring the point back down (i.e. set **r<3**).

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/how_4.PNG"
    width="500px">

This cycle repeats, with the radius **r** coming closer and closer to **r = 3** 
    each time, until eventually the point settles around **r = 3**.

### Side note, how did you convert from polar to cartesian coordinates?
---
I'll omit the proof for now (you can read about it most chaos theory textbooks),
    using the following set of equations:

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/polar_coords_conversion.PNG"
    width="200px">

We can substitute in our rdot and theta dot, rearrange and use simultaneous
    equations to get the system of equations in cartesian coordinates.

We can also convert back to polar coordinates using the following identities:

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/cartesian_coords_conversion.PNG"
    width="200px">

## What series of values of r will be the **best**?
Now we get into the real guts of the problem, what values of r do I set in order
    to **optimally** reach the target radius of r = 3?

As you may have guessed, the solution involves some form of **optimisation**.

As such, let's define the various elements of the problem.

### The cost function
---

For those that don't know, the cost function is a way we can tell the optimiser
    how "*wrong*" the solution is.

We can also inform the optimiser of how their answer should change in order to
    be more correct by computing the derivative of the loss function.

So... How do we quantify loss?

Well there are in fact two sources of loss in our system.

1. How "*off*" from the target state the current state is. We will refer to this
    kind of loss as **state loss**.

2. How much the control input deviates from the rest position (i.e. there is 
    some cost associated with changing the position of the ring of stability).
    We will refer to this kind of loss as **control loss**.

### State Loss
---

Starting off with the **state loss**, we can graph the change in the state 
    position against time. (NOTE: **x0, x1, ..., xN** from this point forward 
    shall refer to the **EVALUATION STATE** of the system, DO NOT CONFUSE IT 
    FOR **x** in cartesian space).

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/cost_1.PNG"
    width="500px">

Now we have to somehow put this into a computer. The easiest way to do this is
    through **euler discretisation**.

Euler discretisation entails determining the gradient at a point (i.e. figuring
 out the direction of the system's vector field), and then advancing some
 finite timestep forward.

 We also know that the derivative of the system is influenced by the 
    **control state** (u0). As such, we will denote the deterivative of the 
    system at evalutation state xN, under the influence of control state uN as:

 <img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/cost_3.PNG"
    width="250px">

Graphically speaking, this means that if we wanted to determine the evaluation 
    state **x1** from **x0** under the influence of control state **u0**, for a 
    timestep **deltaT**, we'd end up with the following:

 <img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/cost_2.PNG"
    width="500px">

Great, how does this relate to the cost function though?

Well we really want to get our evaluation state as close to xr = 3 as possible,
hence we can define cost using the following equation:

 <img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/cost_5.PNG"
    width="250px">

where || ... ||^2 denotes the **euclidean norm** (or 2-norm).

What this looks like graphically is as follows:

 <img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/cost_4.PNG"
    width="500px">

This by itself is not the cost function, but the **running cost**. In order to 
    the cost function, we'll need to sum all the running costs over the entire
    **control horizon** (which is the finite number of timesteps we predict
    into the future). This idea is captured by the following equation:

 <img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/cost_6.PNG"
    width="250px">

And graphically interpreted like so:

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/cost_7.jpg"
    width="500px">

### Control loss
---
We can apply the same logic to the control loss. Applying the same logic as we
    did for the state loss, we arrive at the graphical representation of the
    control loss.

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/cost_8.PNG"
    width="500px">

Now, if we combine both the state and control loss into a single sum, we get
    the **System loss**, which is denoted by the following equation:

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/cost_9.PNG"
    width="340px">

You'll also notice that the state and control loss have a **Q** and **R** term
    appended to their respective 2-norms.

These are simply weights (real numbers) that you can specify to bias the 
    optimiser into prioritising one loss over the other.

For instance, if control loss is more expensive than state loss, then the 
    optimiser would priorities a **smoother** optimal trajectory.

In contrast, if state loss is more expensive than control loss, me might see a
    **sharper** change in the state trajectory.

Graphically the two would look like so:

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/cost_10.PNG"
    width="500px">

### Constraints
---
We'll get into those a bit later, keep reading ;)

## Control and state vectors

Now we can move onto how the controls and states will be represented in
    computer memory. Starting off with controls, we have:

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/state_1.PNG"
    width="200px">

And from these controls, we can determine the various evaluation states:

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/state_3.PNG"
    width="340px">

Which is where you might find yourself running into an intutition problem
    (much like [I did](https://math.stackexchange.com/questions/4467882/mpc-trajectory-optimization-intuition)).

It seems like the entries of the state vector (excluding X0), depend on the
    previous entries.

This makes the problem conceptually difficult to optimise. For instance, if
    we were to change the control input u1, then all the evaluation states 
    after it would also change.

Despite this, it is relatively straight forward to optimise XT. Conceptually,
    you'd evaluate each of the entries, and then through finite difference or
    some other method, determine the descent direction of the vector.

Thus it is possible, but would be very tedious to do manually. And thus, we get
    into a very interesting field in applied math / computer science

## Finite difference and symbolic differentiation

There are a couple of ways of computing derivatives in computers. Two prominant
    methods are:

- Finite difference
- Symbolic

### Finite difference
---
The finite difference formula comes from the fundimental theorem of calculus 
    (the proofs we all take for granted in highschool).

It's purpose was to demonstrate that the derivative of a function was the 
    difference of the function at a point **x**, and a point **x + h*ei**

Where **h** is some small stepsize (typically 10e-5), and ei is the unit vector
    along axis **i** in some N-dimensional space

 <img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/autodiv_7.PNG"
    width="200px">

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/autodiv_4.PNG"
    width="500px">

If h is taken to an infinitely small value (i.e. h -> 0), then the formula above
    produces a scalar value representing the tangent at point x.

So great! let's just use that right?

Well, if the step size (h) is a finite size, then for every step we will accumulate
    **truncation error** (those who've used the euler method will understand
    what this is).

More trouble comes when we try to implement this method in a computer. You
    see, for very small numbers, floating point arithmetic is used.

Just like how we use bits to count up, we can use them to count down.

For instance, one bit position could represent 1/2, the next 1/4, and so on

Generally speaking, the further right of the decimal place you go, the value represented by that bit equals the following formula:

```
1/2^N.
```

A popular youtuber [Mental Outlaw](https://www.youtube.com/watch?v=WJgLKO-qac0)
    does a good job explaining this in one of his videos. I highly recommend you
    give it a watch.

Because computing these numbers is a pain, we often round the answer to some
    specific number of significant figures.

Graphically speaking, this rounding error and step size (truncation) 
    error results
    in the finite difference method having to deal with two forms of inversely
    correlated error:

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/autodiv_5.PNG"
    width="500px">

For MPC applications that require absolute precision (like say a robot arm 
    dealing with brain surgery, looking at you neuralink), this error is 
    unacceptable.

### Symbolic differentiation
---
Symbolic differentiation won't accumulate nearly as much error as finite 
    difference, but suffers from the same problem anyone who's had to derive 
    a university level equation has encountered.

Known as **expression swell**, derivative expressions become exponentially 
    larger with higher iterations.

For example, let's consider the chain rule for the following equation

```
H(x) = f(x)g(x)
H'(x) = f'(x)g(x) + f(x)g'(x)
H''(x) = f''(x)g(x) + f'(x)g'(x) + f'(x)g'(x) + f(x)g''(x)
...
```

Here you can see how H(x) has 2 evaluations, H'(x) has 4 evaluations, H''(x)
    has 8 and so on.

Other examples include derivatives of functionals, like the soft ReLU function
    you see below.

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/autodiv_6.PNG"
    width="500px">

While simple on the outside, its derivative quickly blows up into something
    conceptually difficult to interpret.

Yet, somehow, companies like Google, Tesla, SpaceX and more, leverage neural
    networks with **BILLIONS** of parameters!

As we'll come to see in the next bit, a specific idea in the applied math
    and computer science field is one of the core factors that allowed the
    ML boom to happen.

## Automatic differentiation and 3b1b

The famous youtuber, [3b1b](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
    (an acronym for "three blue one brown" in reference to the authors iris
    phenotype), put out a couple videos explaining how backpropagation works
    from the perspective of fundimental calculus.

This was interesting, but for some reason I felt as if it was missing something.
    Specifically the part of **differentiation**.

Neural networks contain millions of parameters that need to be optimised.
    This requires some serious consideration of the computational complexity of
    the optimiser (i.e. how fast it can converge to a solution), as well as
    hardware constraints like memory (try fitting 6 billion neurons into RAM,
    spoilers you can't).

A grossly under-appreciated youtuber, [Sam Sartor](https://www.youtube.com/watch?v=CfAL_cL3SGQ), created a video which gracefully explains what neural networks
    are and aren't, demonstrating that the whole '*decision making ability*'
    of neural networks is simply a series of linear and non-linear transforms 
    of data, done in order to separate said data by some boundary point.

This helped clarify things a bit, but still that whole differentiation thing 
    was a bit of a mystery. How do they do it?

Finally, with the help of [fedja](), I came across [Ari Seff](https://www.youtube.com/watch?v=wG_nF1awSSY), another grossly under-appreciated youtuber who does 
    an absolutely marvelous job explaining what I was missing.

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/autodiv_1.PNG"
    width="600px">

Essentially, the idea is we can break down any given set of equations 
    (or "primals" as its called), into a series of nodes in a tree.

From here, we can work out the individual "tangents" to each node, and thus
    work out the derivative (in a fashion similar to a forward pass
    in a neural network).

The tree above takes two inputs and produces a single scalar output. In practise
    however, you can have as many inputs or outputs as you want.

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/autodiv_2.PNG"
    width="600px">

The really cool thing he points out is the whole idea of a **reverse pass**.

Here you can work out the derivatives of the outputs with respect to each of the
    inputs (either in parallel or individually, otherwise refered to as
    "adjoints").

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/autodiv_3.PNG"
    width="600px">

What this is in an abstract sense, reverse mode automatic differentiation is 
    a generalised backpropagation method.

His video even goes into why reverse mode is prefered over forward mode for
    neural network applications, and some memory optimisations used to
    allow multi-billion parameter neural networks to be optimised
    (like performing a matrix vector product as a series of vector-vector 
    products in order to optimise memory usage).

## Automatic differentiation key takeaways

With auto-diff, we can thus define any loss function we want (with 
    approved primals) and have an auto-diff library handle the creation of 
    these "trees".

Indeed, this is also what popular packages like [CasADi](https://web.casadi.org/)
    do (It's what the "AD" in the title stands for).

This also means things like **constraints** can be added into your loss 
    function with `if statements`.

## The results?

Assuing you have `JAX` (An automatic differentiation research project by 
    Google I believe) installed, running the following command will yield the
    following output.

```
python3 shooting_method.py
```

```
Cmd-line output:

INFO:MPC Shooting method JAX:50 epoch - Loss: 34.95061111450195
INFO:MPC Shooting method JAX:100 epoch - Loss: 34.5342903137207
INFO:MPC Shooting method JAX:150 epoch - Loss: 34.19135665893555
INFO:MPC Shooting method JAX:200 epoch - Loss: 33.9622802734375
INFO:MPC Shooting method JAX:250 epoch - Loss: 33.81520462036133
INFO:MPC Shooting method JAX:300 epoch - Loss: 33.61939239501953
INFO:MPC Shooting method JAX:350 epoch - Loss: 33.46249008178711
INFO:MPC Shooting method JAX:400 epoch - Loss: 33.20464324951172
INFO:MPC Shooting method JAX:450 epoch - Loss: 32.96542739868164
INFO:MPC Shooting method JAX:500 epoch - Loss: 32.70095443725586
INFO:MPC Shooting method JAX:550 epoch - Loss: 32.40840530395508
INFO:MPC Shooting method JAX:600 epoch - Loss: 32.114013671875
INFO:MPC Shooting method JAX:650 epoch - Loss: 31.766572952270508
INFO:MPC Shooting method JAX:700 epoch - Loss: 31.380861282348633
INFO:MPC Shooting method JAX:750 epoch - Loss: 31.044452667236328
INFO:MPC Shooting method JAX:800 epoch - Loss: 30.659595489501953
INFO:MPC Shooting method JAX:850 epoch - Loss: 30.212190628051758
```

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/loss_1.PNG"
    width="600px">

As you can see, we can indeed optimise this loss_function we've made ourselves,
    using something simple like **Stochastic Gradient Descent** (I didn't 
    bother with second order methods since the aim was to understand MPC
    conceptually).

The Q and R values were set to 3.0 and 0.1 respectively (putting more cost on
    the state than the control loss).

## The confession
Hopefully now you understand MPC a bit better conceptually, however I have been
lying to you this entire time.

What I have been showing you isn't exactly MPC. It is but one method of 
    implementing it. Specifically, the above implementation is called
    The **shooting method**.

This method aims to optimise the problem by changing the control variables.
    As you can see, after 850 epochs, it isn't much different to the 
    uncontrolled trajectory.

I know, but I can make up for it. There is another method I have to show you,
    which produces much nicer results (but has some interesting quirks of its
    own).

## The (direct) **Collocation method**

Rather than optimise over the control parameters and compute the evaluation
    state from it, why not optimise over the state parameters, and through
    an inverse relation, *recover* the controls that would have resulted in 
    state trajectory.

That's certainly a mouthful, so I'll draw you a picture about what I mean. 
    
<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/coll_1.PNG"
    width="600px">

This is the original euler discretisation scheme we were using for the 
    shooting method. It requires the knowledge of point xi and control ui.

However, what if we were to rearrange the equation like so:

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/coll_2.PNG"
    width="600px">

Now if we knew points xi+1 and xi, we could work out the derivative from
    xi to xi+1 (the simple rise / run rule we learned in early middle school).

We could then figure out the inverse of our equation for xdot (i.e. rearrange
    the equation and solve for our control variable lambda).

This is the essence of the **collocation method**, we optimise our trajectory,
    and through an inverse function, work out the controls that would have 
    resulted in said trajectory.

Running the following command will yield the following output:

```
python3 collocation_method.py
```

```
cmd-line output:
INFO:MPC Collocation method JAX:50 epoch - Loss: 51.6541633605957
INFO:MPC Collocation method JAX:100 epoch - Loss: 112.57476806640625
INFO:MPC Collocation method JAX:150 epoch - Loss: 81.4659652709961
INFO:MPC Collocation method JAX:200 epoch - Loss: 59.191410064697266
INFO:MPC Collocation method JAX:250 epoch - Loss: 40.286216735839844
INFO:MPC Collocation method JAX:300 epoch - Loss: 28.314748764038086
INFO:MPC Collocation method JAX:350 epoch - Loss: 25.434396743774414
INFO:MPC Collocation method JAX:400 epoch - Loss: 24.510356903076172
INFO:MPC Collocation method JAX:450 epoch - Loss: 24.55760955810547
INFO:MPC Collocation method JAX:500 epoch - Loss: 23.863821029663086
```

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/coll_3.PNG"
    width="600px">

Which produces a much better response than the shooting method in far fewer 
    epochs.

## Comparing the shooting and collocation methods

There are a couple of key comparisons between the shooting and collocation 
    methods that are important to understand

### Shooting method

The shooting method can be summarised by the following equation:

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/comp_1.PNG"
    width="400px">

The following points define its behaviour

- Optimise over **controls**
- State trajectory is **implicit** (meaning you don't have to formally state it)
- Dynamics is an implicit constraint (always satisfied)
- Applying constraints difficult

###  (Direct) Collocation method

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/comp_2.PNG"
    width="400px">

- Optimise over **states**
- Controls and forces are implicit
- Dynamics is an **explicit** constraint (can be soft)
- Applying constraints easy

## (Finally) onto constaints

The key takeaway is while the collocation method is generally faster, the 
    dynamics of the system are not an implicit constraint.

A good analogy is if this were a simulation of the real world, then it means 
    the method can possibly violate the laws of physics if you don't specify
    your constraints correctly (for example the optimal trajectory with the 
    least loss might actually put the robot on a trajectory through a wall!)

The university of [Berkley](https://www.youtube.com/watch?v=pBCVQbZtv78&list=PLIALWIqZVSKgtiCQaKOFJ3UwRTZ_55X55) provides a great guest lecture on this
    subject, demonstrating how the method can be difficult to work with when
    considering intricate interactions like a robot hand catching and 
    manipulating a ball (sometimes the fingers can phase through the ball, and 
    articulate in ways that wouldn't be possible normally).

Sometimes this notion of soft constraints can be useful.

A good analogy is a race track. Is it better to slow down and perform a turn at
    a bend in the track, or is it better to cut the corner and drive over the 
    grass?

<img src="https://storage.googleapis.com/starfighter-public-bucket/wiki_images/resume_photos/MPCDemo/corner_cutting.jpg"
    width="400px">

### When to use hard and soft constraints

For some applications (like robots trying to climb a platform or something),
    soft and hard constraints must be used in tandem.

For example, when taking a step, the area around the robots leg must be a hard
    constraint (as trying to walk through one of your limbs will produce 
    unfavourable results).

Another thing to look out for is **deadlock**. Deadlock occurs when both the 
    control and state parameters have hard constraints, and the optimiser 
    reaches a point where it can't descend any direction without violating
    the constraints.

To avoid this phenomenon, adhere to the following rules


| State Constraints      | Control Constraints        | Combination |
| :--------------:  | :----------------:    | :------------: |
| Soft              |      Soft             |   OK
| Hard              |      Soft             |   OK          |
| Hard              |      Hard             |   DEADLOCK    |
| Soft              |      Hard             |   OK

## Downsides of MPC
The only major downside of MPC is that in order to make it work, you need a
    dynamical system representation of the thing you want to control.

That can be very difficult to do, even for seasoned mathematicians.

MPC does somewhat compensate for this, as after each prediction, MPC only
    executes the first step of the prediction. If the model is somewhat off,
    this gives MPC a chance to account for that.

## Closing statement
There's a lot more to MPC than what I've lead on in this example, but I hope
    you learned something and enjoyed reading through this project as much
    as I did making it :)

But what I have said does cover the fundimentals of how things like Space X's
    rockets land themselves (the control system part anyway, the 
    dynamical representation part is a whole other story).

Well, that's all I have to say, so until next time

- Despicable-Bee


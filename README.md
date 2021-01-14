# Installation

This will get Nacime’s branch of firedrake 
(from the [pointwise adjoint operator PR](https://github.com/firedrakeproject/firedrake/pull/1674))
with volumential and meshmode installed.
```bash
curl -O https://raw.githubusercontent.com/benSepanski/firedrake/pointwise-adjoint-operator-layer-potentials/scripts/firedrake-install
python3 firedrake-install --nonlocal-operator-dependencies --package-branch firedrake pointwise-adjoint-operator --package-branch PyOP2 DataCarrier-object-versionning --package-branch pyadjoint pointwise-adjoint-operator --package-branch tsfc pointwise-operators --package-branch ufl external-operator
source firedrake/bin/activate
```
While we wait for a few loopy pulls to go through, we should also grab my branch of loopy
(right now, meshmode works with a few very small fixes to firedrake's loopy branch).
```bash
cd firedrake/src/loopy
git remote add sepanski_fork https://github.com/benSepanski/loopy.git
git fetch sepanski_fork firedrake-usable_for_potentials
git checkout firedrake-usable_for_potentials

cd ..
pip install -e loopy/
```
Next, we hack master loopy into the firedrake environment as loopyy and make volumential
(and some other relevant packages) look for loopyy
(from the master branch) instead of loopy (from firedrake).
```bash
pip install git+https://gitlab.tiker.net/ben_sepanski/loopy.git@loopy_to_loopyy#egg=loo.pyy
for i in boxtree sumpy volumential ; do
    pip install --upgrade git+https://gitlab.tiker.net/ben_sepanski/$i.git@loopy_to_loopyy ;
done ;
```
Finally, go and checkout the correct branch of firedrake
```bash
cd firedrake
git remote add sepanski_fork https://github.com/benSepanski/firedrake.git
git fetch sepanski_fork pointwise-adjoint-operator-layer-potentials
git checkout --track sepanski_fork/pointwise-adjoint-operator-layer-potentials

cd ..
pip install -e firedrake
```

Now go to whichever directory you want this github project in, clone it,
and try running `python3 volumential_test.py`. You should get an error
of about `1e-03`.

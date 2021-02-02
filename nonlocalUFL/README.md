# Directory Layout


# Installation

This will get the pointwise-adjoint-operator-layer-potentials branch of firedrake 
(branched off of the [pointwise adjoint operator PR](https://github.com/firedrakeproject/firedrake/pull/1674))
with volumential and meshmode installed.
We also install [pyamg](https://github.com/pyamg/pyamg).
```bash
curl -O https://raw.githubusercontent.com/benSepanski/firedrake/pointwise-adjoint-operator-layer-potentials/scripts/firedrake-install
python3 firedrake-install --complex --nonlocal-operator-dependencies --package-branch firedrake pointwise-adjoint-operator-layer-potentials --package-branch PyOP2 DataCarrier-object-versionning --package-branch pyadjoint pointwise-adjoint-operator --package-branch tsfc pointwise-operators --package-branch ufl external-operator
source firedrake/bin/activate
pip install pyamg
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
# update modepy
pip install --upgrade git+https://gitlab.tiker.net/inducer/modepy.git ;
# get loo.pyy and some things that use loo.pyy
pip install git+https://gitlab.tiker.net/ben_sepanski/loopy.git@loopy_to_loopyy#egg=loo.pyy ;
for i in boxtree sumpy meshmode ; do
    pip install --upgrade -U "eager" git+https://gitlab.tiker.net/ben_sepanski/$i.git@loopy_to_loopyy ;
done ;
# Need multiple versions of boxtree also
pip install git+https://gitlab.tiker.net/ben_sepanski/boxtree.git@boxtree_to_boxtreee#egg=boxtreee ;
pip install --upgrade -U "eager" git+https://gitlab.tiker.net/ben_sepanski/pytential.git@loopy_to_loopyy_boxtree_to_boxtreee ;
# Now grab volumential
pip install --upgrade -U "eager" git+https://gitlab.tiker.net/ben_sepanski/volumential.git@loopy_to_loopyy ;
```
Run `pip list | grep loo`. If you see both `loo.py` and `loopy`,
run `pip uninstall loopy`. Run `pip list | grep loo` again, and you should
no longer see `loopy`.

Now go to whichever directory you want this github project in, clone it,
and try running `python3 volumential_test.py`. You should get an error
of about `1e-03`.

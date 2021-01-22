.PHONY: meshes clean

meshes:
	@echo "    Building meshes"
	@python3 bin/make_meshes example
	@echo "	   meshes built"

clean:
	@echo "    Removing HelmholtzSommerfeldProblem/meshes/*.geo files"
	rm -rf HelmholtzSommerfeldProblem/meshes/*.geo
	@echo "    Removing HelmholtzSommerfeldProblem/meshes/*.step files"
	rm -rf HelmholtzSommerfeldProblem/meshes/*.step
	@echo "    Removing HelmholtzSommerfeldProblem/meshes/*.msh files"
	rm -rf HelmholtzSommerfeldProblem/meshes/*.msh
	@echo "    Removing HelmholtzSommerfeldProblem/meshes/brep_files/*.brep files"
	rm -rf HelmholtzSommerfeldProblem/meshes/brep_files/*.msh

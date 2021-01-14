.PHONY: meshes

meshes:
	@echo "    Building meshes"
	@python3 bin/make_meshes example
	@echo "	   meshes built"

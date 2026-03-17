idf_component_register(
    SRCS
        "main.c"
        "dense_c.c"
        "dense_pie.c"
        "dense_pie.S"
        "sparse_bitmask.c"
        "sparse_pie.c"
        "sparse_pie.S"
        "matrix_utils.c"
    INCLUDE_DIRS "."
) 
FUNCTION reduce_matrices(n) RESULT(acc)
    IMPLICIT NONE

    INTEGER, INTENT(IN) :: n
    REAL(kind=8), DIMENSION(3, 3, n) :: matrices
    REAL(kind=8), DIMENSION(3, 3) :: acc

    INTEGER :: i

    CALL srand(1337)
    matrices = rand()

    acc = 0

    !$OMP PARALLEL DO REDUCTION(+:acc)
    DO i = 1, n
        acc = acc + matrices(1:3, 1:3, i)
    ENDDO

END FUNCTION

SUBROUTINE print_usage_message()
    IMPLICIT NONE

    PRINT*, "Parâmetros incorretos. Uso:"
    PRINT*, "  main <NUM>"
    PRINT*, "onde:"
    PRINT*, "  <NUM>      Tamanho do vetor de matrizes 3x3."
END SUBROUTINE

PROGRAM exemplo_matriz
    IMPLICIT NONE
    INTERFACE
        FUNCTION reduce_matrices(n)
            IMPLICIT NONE
            INTEGER, INTENT(IN) :: n
            REAL(kind=8), DIMENSION(3, 3) :: reduce_matrices
        END FUNCTION
    END INTERFACE


    CHARACTER(len=32) :: number_str
    INTEGER :: n, stat
    REAL(kind=8), DIMENSION(3, 3) :: acc

    IF (iargc() /= 1) THEN
        CALL print_usage_message()
        RETURN
    ENDIF

    CALL getarg(1, number_str)
    READ(number_str,*, iostat=stat) n

    acc = reduce_matrices(n)

    PRINT*, "Resultado: ", acc

END PROGRAM exemplo_matriz

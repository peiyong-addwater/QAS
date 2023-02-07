using Quantikz
circuit_vqls = [
                H(1), H(2), H(3), H(4),
                U("Rot(\\phi_0, \\theta_0, \\omega_0)",1),
                U("Rot(\\phi_1, \\theta_1, \\omega_1)",2),
                U("Rot(\\phi_2, \\theta_2, \\omega_2)",3),
                U("Rot(\\phi_3, \\theta_3, \\omega_3)",4),
                CNOT(2,1), CNOT(3,4),
                U("Rot(\\phi_4, \\theta_4, \\omega_4)",2),
                CNOT(2,1),
                U("Rot(\\phi_5, \\theta_5, \\omega_5)",1),
                U("Rot(\\phi_6, \\theta_6, \\omega_6)",2),
               ]
savecircuit(circuit_vqls, "vqls_circuit.pdf")

circuit_h2 = [
                U("Rot(\\phi_0, \\theta_0, \\omega_0)",1),
                U("Rot(\\phi_1, \\theta_1, \\omega_1)",3),
                CNOT(3,4),
                U("Rot(\\phi_2, \\theta_2, \\omega_2)",3),
                U("Rot(\\phi_3, \\theta_3, \\omega_3)",4),
                CNOT(3,2), CNOT(1,2), CNOT(3,4),
                U("Rot(\\phi_4, \\theta_4, \\omega_4)",1),
                U("Rot(\\phi_5, \\theta_5, \\omega_5)",3),
                CNOT(2,1),CNOT(4,3),CNOT(1,2),
                U("Rot(\\phi_6, \\theta_6, \\omega_6)",3),
                CNOT(2,1),CNOT(4,3),CNOT(3,2),
                U("Rot(\\phi_7, \\theta_7, \\omega_7)",2),
                CNOT(3,2),CNOT(1,2),CNOT(4,3),
                U("Rot(\\phi_8, \\theta_8, \\omega_8)",3),
             ]
savecircuit(circuit_h2, "h2_circuit.pdf")

circuit_h2o = [
                U("Rot(\\phi_0, \\theta_0, \\omega_0)",3),
                U("Rot(\\phi_1, \\theta_1, \\omega_1)",5),
                U("Rot(\\phi_2, \\theta_2, \\omega_2)",8),
                CNOT(3,2),CNOT(5,6),CNOT(8,7),
                U("Rot(\\phi_3, \\theta_3, \\omega_3)",2),
                U("Rot(\\phi_4, \\theta_4, \\omega_4)",3),
                CNOT(6,5),
                U("Rot(\\phi_5, \\theta_5, \\omega_5)",8),
                CNOT(2,1),CNOT(3,4),
                U("Rot(\\phi_6, \\theta_6, \\omega_6)",5),
                CNOT(8,7),
                U("Rot(\\phi_7, \\theta_7, \\omega_7)",1),
                CNOT(3,2),
                U("Rot(\\phi_8, \\theta_8, \\omega_8)",4),
                U("Rot(\\phi_9, \\theta_9, \\omega_9)",7),
                CNOT(2,1),CNOT(8,7),
                U("Rot(\\phi_{10}, \\theta_{10}, \\omega_{10})",1),
                U("Rot(\\phi_{11}, \\theta_{11}, \\omega_{11})",2),
                CNOT(7,6),CNOT(6,5),
                U("Rot(\\phi_{12}, \\theta_{12}, \\omega_{12})",7),
                U("Rot(\\phi_{13}, \\theta_{13}, \\omega_{13})",6),
                CNOT(8,7),CNOT(5,6),CNOT(7,8),
                U("Rot(\\phi_{14}, \\theta_{14}, \\omega_{14})",6),
                U("Rot(\\phi_{15}, \\theta_{15}, \\omega_{15})",7),
                U("Rot(\\phi_{16}, \\theta_{16}, \\omega_{16})",8),
                CNOT(5,6),CNOT(4,5),
                U("Rot(\\phi_{17}, \\theta_{17}, \\omega_{17})",6),
                CNOT(3,4),
                U("Rot(\\phi_{18}, \\theta_{18}, \\omega_{18})",5),
                U("Rot(\\phi_{19}, \\theta_{19}, \\omega_{19})",3)
              ]

savecircuit(circuit_h2o, "h2o_circuit.pdf")

circuit_lih = [
                U("Rot(\\phi_0, \\theta_0, \\omega_0)",1),
                U("Rot(\\phi_1, \\theta_1, \\omega_1)",2),
                U("Rot(\\phi_2, \\theta_2, \\omega_2)",3),
                U("Rot(\\phi_3, \\theta_3, \\omega_3)",5),
                U("Rot(\\phi_4, \\theta_4, \\omega_4)",6),
                U("Rot(\\phi_5, \\theta_5, \\omega_5)",8),
                U("Rot(\\phi_6, \\theta_6, \\omega_6)",10),
                CNOT(5,4),CNOT(3,4),CNOT(4,3),CNOT(3,2),
                U("Rot(\\phi_7, \\theta_7, \\omega_7)",4),
                CNOT(3,4)
              ]

savecircuit(circuit_lih, "lih_circuit.pdf")

circuit_maxcut_1 = [
                      H(1),H(2),H(3),H(4),H(5),H(6),H(7),
                      U("Rot(\\phi_0, \\theta_0, \\omega_0)",1),
                      U("Rot(\\phi_1, \\theta_1, \\omega_1)",2),
                      U("Rot(\\phi_2, \\theta_2, \\omega_2)",3),
                      U("Rot(\\phi_3, \\theta_3, \\omega_3)",5),
                      U("Rot(\\phi_4, \\theta_4, \\omega_4)",6),
                      U("Rot(\\phi_5, \\theta_5, \\omega_5)",7),
                      CNOT(3,4),
                      U("Rot(\\phi_6, \\theta_6, \\omega_6)",3),
                      U("Rot(\\phi_7, \\theta_7, \\omega_7)",4),
                      CNOT(4,3),CNOT(3,4)
                   ]

savecircuit(circuit_maxcut_1, "maxcut_circuit_1.pdf")

circuit_maxcut_2 = [
                    H(1),H(2),H(3),H(4),H(5),H(6),H(7),
                    U("Rot(\\phi_0, \\theta_0, \\omega_0)",2),
                    U("Rot(\\phi_1, \\theta_1, \\omega_1)",7),
                    CNOT(7,1),
                    U("Rot(\\phi_2, \\theta_2, \\omega_2)",1),
                    U("Rot(\\phi_3, \\theta_3, \\omega_3)",3),
                    U("Rot(\\phi_4, \\theta_4, \\omega_4)",4),
                    U("Rot(\\phi_5, \\theta_5, \\omega_5)",5),
                    U("Rot(\\phi_6, \\theta_6, \\omega_6)",6),
                    CNOT(5,6),CNOT(6,5),CNOT(7,6),CNOT(6,7),
                    U("Rot(\\phi_7, \\theta_7, \\omega_7)",6)
                   ]

savecircuit(circuit_maxcut_2, "maxcut_circuit_2.pdf")

circuit_weighted_maxcut = [
                           H(1),H(2),H(3),H(4),H(5),
                           U("Rot(\\phi_0, \\theta_0, \\omega_0)",5),
                           CNOT(5,1),
                           U("Rot(\\phi_1, \\theta_1, \\omega_1)",1),
                           U("Rot(\\phi_2, \\theta_2, \\omega_2)",2),
                           U("Rot(\\phi_3, \\theta_3, \\omega_3)",5),
                           CNOT(5,1),CNOT(2,3),
                           U("Rot(\\phi_4, \\theta_4, \\omega_4)",4),
                           U("Rot(\\phi_5, \\theta_5, \\omega_5)",3)
                          ]
savecircuit(circuit_weighted_maxcut, "weighted_maxcut_circuit.pdf")

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





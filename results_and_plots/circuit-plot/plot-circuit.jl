using Quantikz
circuit_vqls = [
                H(1), H(2), H(3), H(4),
                U("Rot(\\phi_0, \\theta_0, \\omega_0)",1),
                U("Rot(\\phi_1, \\theta_1, \\omega_1)",2),
                U("Rot(\\phi_2, \\theta_2, \\omega_2)",3),
                U("Rot(\\phi_3, \\theta_3, \\omega_3)",4)
               ]
savecircuit(circuit_vqls, "vqls_circuit.pdf")

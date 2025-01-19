from manim import *

class RepeatKVScene(Scene):
    def construct(self):
        # Initial tensor
        initial_tensor = Matrix([[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]])
        initial_tensor.move_to(LEFT * 4)  # Position on left side
        
        # Show initial tensor and its shape
        shape1 = Text("[2, 2, 3, 2]", font_size=24).next_to(initial_tensor, DOWN)
        step1 = Text("1. Original tensor", font_size=24).to_edge(UP)
        self.play(Write(initial_tensor), Write(shape1), Write(step1))
        self.wait()

        # Expanded tensor (centered)
        expanded_tensor = Matrix([
            [[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]],
            [[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]]
        ])
        expanded_tensor.move_to(ORIGIN)  # Position in center
        
        shape2 = Text("[2, 2, 2, 3, 2]", font_size=24).next_to(expanded_tensor, DOWN)
        step2 = Text("2. Add dimension and expand", font_size=24).to_edge(UP)
        self.play(
            FadeOut(step1),
            FadeIn(step2),
            FadeIn(expanded_tensor),
            FadeIn(shape2)
        )
        self.wait()

        # Final tensor (right side)
        final_tensor = Matrix([
            [["a", "b"], ["c", "d"], ["a", "b"], ["c", "d"]],
            [["e", "f"], ["g", "h"], ["e", "f"], ["g", "h"]]
        ])
        final_tensor.move_to(RIGHT * 4)  # Position on right side
        
        shape3 = Text("[2, 4, 3, 2]", font_size=24).next_to(final_tensor, DOWN)
        step3 = Text("3. Reshape to combine dimensions", font_size=24).to_edge(UP)
        self.play(
            FadeOut(step2),
            FadeIn(step3),
            FadeIn(final_tensor),
            FadeIn(shape3)
        )
        self.wait(2)
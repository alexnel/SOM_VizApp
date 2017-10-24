/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package barebone;

/**
 *
 * @author alexandra
 */
import javax.swing.*;
import java.awt.*;
import java.util.Random;

public class GridDisplay extends JPanel {
    
    public static final Color CITY = new Color(214,217,223);

    public static final Color[] TERRAIN = {
        Color.BLACK,
        new Color(51,0,0),
        new Color(102,0,0),
        new Color(153,0,0),
        new Color(204,0,0),
        new Color(255,0,0),
        new Color(255,51,51),
        new Color(255,102,102),
        new Color(255,153,153),
        new Color(255,204,204),
        Color.WHITE
    };

    int NUM_ROWS;
    int NUM_COLS;

    public static final int PREFERRED_GRID_SIZE_PIXELS = 20;

    // In reality you will probably want a class here to represent a map tile,
    // which will include things like dimensions, color, properties in the
    // game world.  Keeping simple just to illustrate.
    private final Color[][] terrainGrid;

    public GridDisplay(int[][] input, int row, int col){
        NUM_ROWS = row;
        NUM_COLS = col;
        
        this.terrainGrid = new Color[NUM_ROWS][NUM_COLS];
        
        for (int i = 0; i < NUM_ROWS; i++) {
            for (int j = 0; j < NUM_COLS; j++) 
            {    
                Color blockColor = TERRAIN[input[i][j]];
                this.terrainGrid[i][j] = blockColor;
            }
        }
        int preferredWidth = NUM_COLS * PREFERRED_GRID_SIZE_PIXELS;
        int preferredHeight = NUM_ROWS * PREFERRED_GRID_SIZE_PIXELS;
        setPreferredSize(new Dimension(preferredWidth, preferredHeight));
    }

    @Override
    public void paintComponent(Graphics g) {
        // Important to call super class method
        super.paintComponent(g);
        // Clear the board
        g.clearRect(0, 0, getWidth(), getHeight());
        // Draw the grid
        int rectWidth = getWidth() / NUM_COLS;
        int rectHeight = getHeight() / NUM_ROWS;

        for (int i = 0; i < NUM_ROWS; i++) {
            for (int j = 0; j < NUM_COLS; j++) {
                // Upper left corner of this terrain rect
                int x = i * rectWidth;
                int y = j * rectHeight;
                Color terrainColor = terrainGrid[i][j];
                g.setColor(terrainColor);
                g.fillRect(x, y, rectWidth, rectHeight);
            }
        }
    }

//    public static void main(String[] args) {
//        // http://docs.oracle.com/javase/tutorial/uiswing/concurrency/initial.html
//        SwingUtilities.invokeLater(new Runnable() {
//            public void run() {
//                JFrame frame = new JFrame("Game");
//                GridDisplay map = new GridDisplay();
//                frame.add(map);
//                frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//                frame.pack();
//                frame.setVisible(true);
//            }
//        });
//    }
}

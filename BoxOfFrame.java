//Interface

import java.awt.*;
import javax.swing.*;
import java.awt.event.*;
import java.util.Scanner;

public class BoxOfFrame
  extends JFrame
  implements ActionListener
{
  private Box thisBox;
  private Ball theBall;
  private Timer timer;
  private JLabel boxL = new JLabel();

  private JFrame box=new JFrame("Balls");

  private JTextField textHeight = new JTextField(10);
  private JTextField textWidth = new JTextField(10);
  private JTextField nCircles = new JTextField(10);
  private JTextField delay = new JTextField(10);
  JLabel result = new JLabel();//in case user doesn't use integers

  private JButton startB = new JButton("Start");
  private JButton quitB= new JButton("Quit");
  private int seconds = 0;

  private int w;//delay


  public BoxOfFrame(int width, int height) {
    //add ball default n. balls
    //theBall=;
    //timer generates occurrences and triggers rewritings and movements
    //now, just put the right buttons in here.


    startB.addActionListener(this);
    quitB.addActionListener(this);
    textHeight.addActionListener(this);
    textWidth.addActionListener(this);
    nCircles.addActionListener(this);
    delay.addActionListener(this);

//    boxL.setPreferredSize(new Dimension(900,400));
//    boxL.setOpaque(true);
//    boxL.setBackground(Color.white);
//    boxL.setForeground(Color.red);
    //boxL.setFont(new Font("Serif", Font.BOLD, 48));
    //boxL.setText("On your marks");
//    boxL.setHorizontalAlignment(JLabel.CENTER);

    JPanel textfields = new JPanel(new GridLayout(4,2));
    textfields.add(new JLabel("Height"));
    textfields.add(textHeight);


    textfields.add(new JLabel("Width"));
    textfields.add(textWidth);


    textfields.add(new JLabel("Number of balls"));
    textfields.add(nCircles);

    textfields.add(new JLabel("Delay"));
    textfields.add(delay);

    this.add(textfields);
    //default values
    textHeight.setText("500");
    textWidth.setText("500");
    nCircles.setText("30");
    delay.setText("20");


    //buttons
    this.add(startB);
    this.add(quitB);

    this.add(result);//just in case user doesn't use integers

    this.setLayout(new FlowLayout()); // enklast...
    this.setPreferredSize(new Dimension(300,400));//only changes size of entire window
    this.pack();//allows full window to be displayed without forcing user to open it themselves
    this.setVisible(true);//obvious
    setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    //NullPointerException
  }

  public void actionPerformed(ActionEvent e) {
    Scanner scHeight = new Scanner(textHeight.getText());
    Scanner scWidth = new Scanner(textWidth.getText());
    Scanner scnCircles = new Scanner(nCircles.getText());
    Scanner scdelay= new Scanner(delay.getText());


    if(e.getSource()==startB){
      if (scHeight.hasNextInt() && scWidth.hasNextInt() && scnCircles.hasNextInt() && scdelay.hasNextInt()) {
        int x = scHeight.nextInt();
        int y = scWidth.nextInt();
        int z=scnCircles.nextInt();
        int w=scdelay.nextInt();//delay

        timer = new Timer(w, this);
        thisBox=new Box(x,y,z,1);
        box.add(thisBox);
        timer.start();
        //this.pack();
        box.pack();
        box.setVisible(true);
      }
      else{
        result.setText("*** ERROR, ALL INPUTS MUST BE INTEGERS ***");

      }


    }



    else if (e.getSource()==quitB){

      System.exit(0);
    }

    //if we are active:
    else if(e.getSource()==timer){
      thisBox.step();

    }


  }


  public static void main(String[] args) {

    BoxOfFrame Bf = new BoxOfFrame(1000,1000);
    //now we want to display the frame of balls

  }
}

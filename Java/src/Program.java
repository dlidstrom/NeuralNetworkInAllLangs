public class Program {
  public static void main(String[] args) {
    System.out.println("Hello, world!");
    Neural neural = new Neural();
    neural.print();
    if (args.length > 0) {
        if (args[0].equals("test")) {
            // load test parameters
        } else if (args[0].equals("production")) {
            // load production parameters
        }
    }
  }
}
